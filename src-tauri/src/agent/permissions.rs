use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum AgentCapability {
    ReadProject,
    ReadSource,
    SearchWiki,
    SearchWeb,
    SearchAnyTxt,
    WriteWiki,
    RunDeepResearch,
    Network,
    Process,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermissionPolicy {
    allowed: Vec<AgentCapability>,
}

impl PermissionPolicy {
    pub fn api_default() -> Self {
        Self {
            allowed: vec![
                AgentCapability::ReadProject,
                AgentCapability::ReadSource,
                AgentCapability::SearchWiki,
                AgentCapability::SearchWeb,
                AgentCapability::SearchAnyTxt,
                AgentCapability::WriteWiki,
                AgentCapability::Network,
            ],
        }
    }

    pub fn allows(&self, capability: AgentCapability) -> bool {
        self.allowed.contains(&capability)
    }

    pub fn require(&self, capability: AgentCapability) -> Result<(), String> {
        if self.allows(capability) {
            Ok(())
        } else {
            Err(format!("Agent capability '{capability:?}' is not allowed"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_default_allows_read_network_and_sandboxed_wiki_writes() {
        let policy = PermissionPolicy::api_default();
        assert!(policy.allows(AgentCapability::SearchWiki));
        assert!(policy.allows(AgentCapability::Network));
        assert!(policy.allows(AgentCapability::WriteWiki));
    }
}
