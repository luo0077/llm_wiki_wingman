use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const MAX_SKILL_BYTES: usize = 24_000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AgentSkill {
    pub name: String,
    pub description: String,
    pub instructions: String,
}

pub fn load_project_skills(project_path: &str, requested: &[String]) -> Vec<AgentSkill> {
    if requested.is_empty() {
        return Vec::new();
    }
    let root = Path::new(project_path).join(".llm-wiki").join("skills");
    requested
        .iter()
        .filter_map(|name| load_one_skill(&root, name))
        .collect()
}

fn load_one_skill(root: &Path, name: &str) -> Option<AgentSkill> {
    let name = normalize_skill_name(name)?;
    let candidates = [
        root.join(format!("{name}.md")),
        root.join(&name).join("SKILL.md"),
    ];
    candidates
        .iter()
        .find_map(|path| load_skill_file(path, &name).ok())
}

fn load_skill_file(path: &PathBuf, fallback_name: &str) -> Result<AgentSkill, String> {
    let meta = fs::metadata(path).map_err(|err| format!("Skill not found: {err}"))?;
    if !meta.is_file() || meta.len() as usize > MAX_SKILL_BYTES {
        return Err("Skill file is not readable or is too large".to_string());
    }
    let raw = fs::read_to_string(path).map_err(|err| format!("Failed to read skill: {err}"))?;
    let (frontmatter, instructions) = split_frontmatter(&raw);
    let name = frontmatter
        .as_deref()
        .and_then(|fm| yaml_string_field(fm, "name"))
        .unwrap_or_else(|| fallback_name.to_string());
    let description = frontmatter
        .as_deref()
        .and_then(|fm| yaml_string_field(fm, "description"))
        .unwrap_or_default();
    Some(AgentSkill {
        name,
        description,
        instructions: instructions.trim().to_string(),
    })
    .filter(|skill| !skill.instructions.is_empty())
    .ok_or_else(|| "Skill instructions are empty".to_string())
}

fn normalize_skill_name(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.contains('/')
        || trimmed.contains('\\')
        || trimmed.contains("..")
    {
        return None;
    }
    Some(trimmed.to_string())
}

fn split_frontmatter(raw: &str) -> (Option<String>, String) {
    let normalized = raw.strip_prefix('\u{feff}').unwrap_or(raw);
    if !normalized.starts_with("---\n") {
        return (None, normalized.to_string());
    }
    let rest = &normalized[4..];
    if let Some(end) = rest.find("\n---") {
        let fm = rest[..end].to_string();
        let after = rest[end + "\n---".len()..]
            .strip_prefix('\n')
            .unwrap_or(&rest[end + "\n---".len()..])
            .to_string();
        (Some(fm), after)
    } else {
        (None, normalized.to_string())
    }
}

fn yaml_string_field(frontmatter: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}:");
    for line in frontmatter.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with(&prefix) {
            continue;
        }
        let value = trimmed[prefix.len()..].trim();
        let value = value
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .or_else(|| value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')))
            .unwrap_or(value);
        if !value.is_empty() {
            return Some(value.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::fs;
    use uuid::Uuid;

    use super::*;

    #[test]
    fn load_project_skills_reads_frontmatter_skill() {
        let root = std::env::temp_dir().join(format!("llm-wiki-skills-{}", Uuid::new_v4()));
        let skills_dir = root.join(".llm-wiki").join("skills");
        fs::create_dir_all(&skills_dir).unwrap();
        fs::write(
            skills_dir.join("reviewer.md"),
            "---\nname: reviewer\ndescription: Review source quality\n---\nCheck claims carefully.",
        )
        .unwrap();

        let skills = load_project_skills(root.to_str().unwrap(), &["reviewer".to_string()]);
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "reviewer");
        assert_eq!(skills[0].description, "Review source quality");
        assert_eq!(skills[0].instructions, "Check claims carefully.");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn load_project_skills_rejects_path_traversal_names() {
        let skills = load_project_skills("/tmp/missing", &["../secret".to_string()]);
        assert!(skills.is_empty());
    }
}
