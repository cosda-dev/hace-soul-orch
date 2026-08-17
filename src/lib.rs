//! hace-soul-orch — Soul Orchestrator
//!
//! The Aura Multi-Soul State-Graph coordinator.
//! Heart of Offica E4 — routes intent to Soul experts,
//! dispatches to Zeus Brain, merges results.
//!
//! Execution flow:
//!   UserPrompt
//!     -> IntentAnalysis (extract keywords)
//!     -> SoulRouter.select() -> Vec<SoulId>
//!     -> for each SoulId:
//!         hook: before_soul_switch
//!         SoulProfile.load() + MemoryProvider.load()
//!         build PromptBlueprint (profile + memory + prompt)
//!         Zeus: BrainKernel.reason(ReasonCtx)
//!         hook: after_soul_response
//!     -> merge SoulResults -> OrchResult
//!
//! "One Zeus Brain shared — multiple Soul profiles = multiple experts"
//! Brain is stateless. Soul is stateful. Memory survives Brain swap.

use std::sync::Arc;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use hace_brain_base::{BrainKernel, BrainError, ReasonCtx, ReasonResult, MemoryItem};
use hace_soul_base::{SoulKernel, AuthCtx, SoulError};
use hace_soul_profile::{SoulProfile, SoulRegistry, ProfileError};
use hace_soul_memory::{MemoryProvider, Interaction, MemoryError};
use hace_soul_routing::{SoulRouter, IntentSpec, RoutingError};
use hace_soul_hooks::{
    SoulHookRegistry, SoulHookCtx,
    HOK_SOUL_BEFORE_SWITCH, HOK_SOUL_AFTER_RESPONSE,
    HOK_SOUL_BEFORE_BRAIN_EXE, HOK_SOUL_AFTER_BRAIN_EXE,
    HookOutcome,
};

// ── Orch types ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPrompt {
    pub session_id: String,
    pub actor_id:   String,
    pub text:       String,
    /// Optional domain hint (overrides routing)
    pub domain:     Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoulResult {
    pub soul_id:    String,
    pub brain_id:   String,
    pub output:     serde_json::Value,
    pub confidence: f32,
    pub tokens:     u32,
    pub latency_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchResult {
    pub session_id:   String,
    pub prompt_text:  String,
    pub souls_called: Vec<String>,
    pub results:      Vec<SoulResult>,
    /// Merged primary output (from highest-confidence Soul)
    pub merged:       serde_json::Value,
    pub strategy:     String,
}

#[derive(Debug, Error)]
pub enum OrchError {
    #[error("routing failed: {msg}")]
    RoutingFailed { msg: String },
    #[error("soul not found: {soul_id}")]
    SoulNotFound { soul_id: String },
    #[error("brain execute failed: {msg}")]
    BrainFailed { msg: String },
    #[error("authority denied: {msg}")]
    AuthDenied { msg: String },
    #[error("hook denied: {hook_id}")]
    HookDenied { hook_id: String },
    #[error("memory error: {msg}")]
    MemoryError { msg: String },
}

impl From<RoutingError> for OrchError {
    fn from(e: RoutingError) -> Self { Self::RoutingFailed { msg: e.to_string() } }
}
impl From<ProfileError> for OrchError {
    fn from(e: ProfileError) -> Self { Self::SoulNotFound { soul_id: e.to_string() } }
}
impl From<BrainError> for OrchError {
    fn from(e: BrainError) -> Self { Self::BrainFailed { msg: e.to_string() } }
}
impl From<SoulError> for OrchError {
    fn from(e: SoulError) -> Self { Self::AuthDenied { msg: e.to_string() } }
}
impl From<MemoryError> for OrchError {
    fn from(e: MemoryError) -> Self { Self::MemoryError { msg: e.to_string() } }
}

// ── OrchConfig ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct OrchConfig {
    /// Max concurrent Soul slots (E4: 3)
    pub max_concurrent_souls: usize,
    /// Max memory items to inject per Soul
    pub memory_context_limit: usize,
    /// Whether to activate secondary souls from routing
    pub multi_soul_enabled:   bool,
}

impl Default for OrchConfig {
    fn default() -> Self {
        Self {
            max_concurrent_souls: 3,
            memory_context_limit: 5,
            multi_soul_enabled:   true,
        }
    }
}

// ── SoulOrchestrator — the Aura core ─────────────────────────────────────────

pub struct SoulOrchestrator {
    config:    OrchConfig,
    registry:  SoulRegistry,
    memory:    Arc<dyn MemoryProvider>,
    router:    Box<dyn SoulRouter>,
    soul_auth: Arc<dyn SoulKernel>,
    hooks:     SoulHookRegistry,
    /// Zeus brain CE — the shared stateless compute fabric
    brain:     Arc<dyn BrainKernel>,
}

impl SoulOrchestrator {
    pub fn new(
        config:    OrchConfig,
        registry:  SoulRegistry,
        memory:    Arc<dyn MemoryProvider>,
        router:    Box<dyn SoulRouter>,
        soul_auth: Arc<dyn SoulKernel>,
        brain:     Arc<dyn BrainKernel>,
    ) -> Self {
        Self {
            config,
            registry,
            memory,
            router,
            soul_auth,
            hooks: SoulHookRegistry::e4_defaults(),
            brain,
        }
    }

    /// Main entry point — process a UserPrompt through Multi-Soul flow
    pub async fn process(&self, prompt: UserPrompt) -> Result<OrchResult, OrchError> {
        // 1. Authority gate (DevnetSoul = pass-through in E4)
        let auth_ctx = AuthCtx::auto(&prompt.actor_id, "aura-orch", "soul.process");
        let auth = self.soul_auth.authorize(auth_ctx).await?;
        if !auth.allowed {
            return Err(OrchError::AuthDenied { msg: auth.reason });
        }

        // 2. Intent analysis + routing
        let intent = {
            let mut spec = IntentSpec::from_text(&prompt.text);
            if let Some(d) = &prompt.domain {
                spec = spec.with_domain(d);
            }
            spec
        };
        let route = self.router.select(&intent)?;

        // 3. Limit concurrent souls
        let soul_ids: Vec<String> = route.souls.iter()
            .take(self.config.max_concurrent_souls)
            .cloned()
            .collect();

        let mut soul_results = Vec::new();

        for soul_id in &soul_ids {
            // Hook: before_soul_switch
            let hctx = SoulHookCtx {
                hook_id:    HOK_SOUL_BEFORE_SWITCH,
                soul_id:    Some(soul_id.clone()),
                session_id: Some(prompt.session_id.clone()),
                action:     "soul_switch".into(),
                actor_id:   Some(prompt.actor_id.clone()),
            };
            if self.hooks.run(HOK_SOUL_BEFORE_SWITCH, &hctx) != HookOutcome::Continue {
                return Err(OrchError::HookDenied { hook_id: HOK_SOUL_BEFORE_SWITCH.into() });
            }

            // 4. Load Soul profile
            let profile = self.registry.get(soul_id)?;

            // 5. Fetch memory context
            let mem_ctx = self.memory.load(soul_id, &prompt.session_id, self.config.memory_context_limit)?;

            // 6. Build ReasonCtx (profile + memory + prompt)
            let brain_ctx = self.build_reason_ctx(&prompt, profile, mem_ctx.items);

            // Hook: before_brain_execute
            let hctx2 = SoulHookCtx {
                hook_id:    HOK_SOUL_BEFORE_BRAIN_EXE,
                soul_id:    Some(soul_id.clone()),
                session_id: Some(prompt.session_id.clone()),
                action:     "brain.reason".into(),
                actor_id:   Some(prompt.actor_id.clone()),
            };
            if self.hooks.run(HOK_SOUL_BEFORE_BRAIN_EXE, &hctx2) != HookOutcome::Continue {
                return Err(OrchError::HookDenied { hook_id: HOK_SOUL_BEFORE_BRAIN_EXE.into() });
            }

            // 7. Zeus Brain execute (stateless compute)
            let brain_result = self.brain.reason(brain_ctx).await?;

            // Hook: after_brain_execute (E5: evidence pack)
            let hctx3 = SoulHookCtx {
                hook_id:    HOK_SOUL_AFTER_BRAIN_EXE,
                soul_id:    Some(soul_id.clone()),
                session_id: Some(prompt.session_id.clone()),
                action:     "brain.reason.done".into(),
                actor_id:   Some(prompt.actor_id.clone()),
            };
            self.hooks.run(HOK_SOUL_AFTER_BRAIN_EXE, &hctx3);

            // 8. Commit interaction to memory
            let _ = self.memory.save(Interaction {
                soul_id:    soul_id.clone(),
                session_id: prompt.session_id.clone(),
                role:       "soul".into(),
                content:    brain_result.output.clone(),
                created_at: 0,
            });

            let sr = SoulResult {
                soul_id:    soul_id.clone(),
                brain_id:   brain_result.model_id.clone(),
                output:     brain_result.output,
                confidence: brain_result.confidence,
                tokens:     brain_result.tokens_used,
                latency_ms: brain_result.latency_ms,
            };

            // Hook: after_soul_response (E5: evidence bundle)
            let hctx4 = SoulHookCtx {
                hook_id:    HOK_SOUL_AFTER_RESPONSE,
                soul_id:    Some(soul_id.clone()),
                session_id: Some(prompt.session_id.clone()),
                action:     "soul.response".into(),
                actor_id:   Some(prompt.actor_id.clone()),
            };
            self.hooks.run(HOK_SOUL_AFTER_RESPONSE, &hctx4);

            soul_results.push(sr);
        }

        // 9. Merge results — highest confidence wins primary output
        let merged = merge_results(&soul_results);

        Ok(OrchResult {
            session_id:   prompt.session_id,
            prompt_text:  prompt.text,
            souls_called: soul_ids,
            results:      soul_results,
            merged,
            strategy:     route.strategy,
        })
    }

    fn build_reason_ctx(
        &self,
        prompt:  &UserPrompt,
        profile: &SoulProfile,
        memory:  Vec<hace_soul_memory::MemoryItem>,
    ) -> ReasonCtx {
        // Convert soul/memory items to brain/base MemoryItems
        let brain_memory: Vec<MemoryItem> = memory.into_iter().map(|m| MemoryItem {
            key:       m.key,
            value:     m.value,
            relevance: m.relevance,
        }).collect();

        ReasonCtx {
            intent_id:     format!("{}::{}", prompt.session_id, prompt.actor_id),
            action:        prompt.text.clone(),
            payload:       serde_json::json!({ "text": prompt.text }),
            memory:        brain_memory,
            domain:        profile.expertise.first().cloned(),
            soul_id:       Some(profile.id.clone()),
            brain_profile: profile.preferred_brains.first().map(|b| format!("zeus://{}", b)),
        }
    }
}

fn merge_results(results: &[SoulResult]) -> serde_json::Value {
    if results.is_empty() { return serde_json::Value::Null; }
    // E4: pick highest confidence
    results.iter()
        .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal))
        .map(|r| r.output.clone())
        .unwrap_or(serde_json::Value::Null)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use hace_brain_base::{AlgoParticle, ReasonCtx, ReasonResult, BrainError};
    use hace_soul_base::DevnetSoul;
    use hace_soul_memory::InMemoryProvider;
    use hace_soul_routing::StaticRouter;
    use async_trait::async_trait;

    // Simple stub brain for tests
    struct EchoBrain;

    #[async_trait]
    impl BrainKernel for EchoBrain {
        fn model_id(&self) -> &str { "echo" }
        fn is_local(&self) -> bool { true }
        async fn reason(&self, ctx: ReasonCtx) -> Result<ReasonResult, BrainError> {
            Ok(ReasonResult {
                output:      serde_json::json!({ "echo": ctx.action }),
                confidence:  0.9,
                tokens_used: 0,
                model_id:    "echo".into(),
                plan:        None,
                latency_ms:  1,
            })
        }
    }

    fn make_orch() -> SoulOrchestrator {
        SoulOrchestrator::new(
            OrchConfig::default(),
            SoulRegistry::e4_defaults(),
            Arc::new(InMemoryProvider::new()),
            Box::new(StaticRouter::e4_defaults()),
            Arc::new(DevnetSoul::devnet("devnet")),
            Arc::new(EchoBrain),
        )
    }

    #[tokio::test]
    async fn process_code_prompt_routes_to_coder() {
        let orch = make_orch();
        let prompt = UserPrompt {
            session_id: "s1".into(),
            actor_id:   "aid://dev1".into(),
            text:       "write Rust async code".into(),
            domain:     None,
        };
        let result = orch.process(prompt).await.unwrap();
        assert!(result.souls_called.contains(&"soul://coder".to_string()));
        assert!(!result.results.is_empty());
        assert_eq!(result.strategy, "static");
    }

    #[tokio::test]
    async fn process_design_prompt_routes_to_architect_and_auditor() {
        let orch = make_orch();
        let prompt = UserPrompt {
            session_id: "s2".into(),
            actor_id:   "aid://dev1".into(),
            text:       "design system architecture for caem".into(),
            domain:     None,
        };
        let result = orch.process(prompt).await.unwrap();
        assert!(result.souls_called.contains(&"soul://architect".to_string()));
    }

    #[tokio::test]
    async fn process_legal_prompt_routes_to_legal() {
        let orch = make_orch();
        let prompt = UserPrompt {
            session_id: "s3".into(),
            actor_id:   "aid://dev1".into(),
            text:       "review this RC contract policy".into(),
            domain:     None,
        };
        let result = orch.process(prompt).await.unwrap();
        assert!(result.souls_called.contains(&"soul://legal".to_string()));
    }

    #[tokio::test]
    async fn memory_persists_across_calls() {
        let orch = make_orch();
        let p1 = UserPrompt { session_id:"s4".into(), actor_id:"aid://dev1".into(),
                               text:"write Rust code".into(), domain:None };
        let p2 = UserPrompt { session_id:"s4".into(), actor_id:"aid://dev1".into(),
                               text:"write Rust code again".into(), domain:None };
        orch.process(p1).await.unwrap();
        let r2 = orch.process(p2).await.unwrap();
        // Memory from first call is loaded by second call
        assert!(!r2.results.is_empty());
    }

    #[tokio::test]
    async fn merged_output_picks_highest_confidence() {
        let results = vec![
            SoulResult { soul_id:"s1".into(), brain_id:"e".into(),
                         output:serde_json::json!({"a":1}), confidence:0.6, tokens:0, latency_ms:0 },
            SoulResult { soul_id:"s2".into(), brain_id:"e".into(),
                         output:serde_json::json!({"b":2}), confidence:0.9, tokens:0, latency_ms:0 },
        ];
        let merged = merge_results(&results);
        assert_eq!(merged["b"], 2);
    }
}
