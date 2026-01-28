//! Tree view types and navigation for the webapp.

use crate::{convert_config, SpotConfig};
use serde::{Deserialize, Serialize};
use solver::tree::{
    Action, ActionTree, ActionTreeNode, Position, Street, TerminalResult,
};

/// State of a node in the tree for display.
#[derive(Debug, Serialize, Deserialize)]
pub struct NodeState {
    /// Node type: "player", "chance", "terminal"
    pub node_type: String,

    /// Current street
    pub street: String,

    /// Player to act (if player node)
    pub player_to_act: Option<usize>,

    /// Player position name (e.g., "BTN", "BB")
    pub player_name: Option<String>,

    /// Current pot size
    pub pot: i32,

    /// Remaining stacks per player
    pub stacks: Vec<i32>,

    /// Available actions (empty for terminal/chance)
    pub actions: Vec<ActionInfo>,

    /// History of actions taken to reach this node
    pub action_history: Vec<ActionHistoryItem>,

    /// Current navigation path
    pub path: String,

    /// Terminal result type (if terminal)
    pub terminal_result: Option<String>,

    /// Error message if navigation failed
    pub error: Option<String>,
}

/// Information about an available action.
#[derive(Debug, Serialize, Deserialize)]
pub struct ActionInfo {
    /// Action index for navigation
    pub index: usize,

    /// Display name (e.g., "Raise to 6")
    pub name: String,

    /// Short name for button (e.g., "R6")
    pub short_name: String,

    /// Action type for styling
    pub action_type: String,

    /// Chip amount (if applicable)
    pub amount: Option<i32>,
}

/// Item in the action history.
#[derive(Debug, Serialize, Deserialize)]
pub struct ActionHistoryItem {
    /// Player who took the action
    pub player: usize,

    /// Player position name
    pub player_name: String,

    /// Action description
    pub action: String,

    /// Path to this point (for jumping back)
    pub path: String,
}

/// Navigate to a node at the given path.
pub fn get_node_at_path_internal(config_json: &str, path: &str) -> NodeState {
    // Parse config
    let spot_config: SpotConfig = match serde_json::from_str(config_json) {
        Ok(c) => c,
        Err(e) => {
            return error_state(&format!("Failed to parse config: {}", e));
        }
    };

    // Convert to TreeConfig
    let tree_config = match convert_config(&spot_config) {
        Ok(c) => c,
        Err(e) => {
            return error_state(&e);
        }
    };

    // Build tree
    let tree = match ActionTree::new(tree_config.clone()) {
        Ok(t) => t,
        Err(e) => {
            return error_state(&e);
        }
    };

    // Navigate to path
    navigate_to_path(&tree, &spot_config, path)
}

/// Get the result of taking an action from the current path.
pub fn get_action_result_internal(
    config_json: &str,
    current_path: &str,
    action_index: usize,
) -> NodeState {
    // Build new path
    let new_path = if current_path.is_empty() {
        action_index.to_string()
    } else {
        format!("{}.{}", current_path, action_index)
    };

    get_node_at_path_internal(config_json, &new_path)
}

fn navigate_to_path(tree: &ActionTree, spot_config: &SpotConfig, path: &str) -> NodeState {
    let num_players = spot_config.num_players;
    let positions = Position::all_for_players(num_players);

    // Parse path indices
    let indices: Vec<usize> = if path.is_empty() {
        Vec::new()
    } else {
        match path
            .split('.')
            .map(|s| s.parse::<usize>())
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(v) => v,
            Err(_) => return error_state(&format!("Invalid path: {}", path)),
        }
    };

    // Track state as we navigate
    let mut current_node = &tree.root;
    let mut action_history = Vec::new();
    let mut current_path = String::new();

    // Initial state
    let mut pot = tree.config.starting_pot();
    let mut stacks: Vec<i32> = (0..num_players)
        .map(|i| tree.config.stack_for_player(i))
        .collect();
    let mut street = Street::Preflop;

    // Deduct blinds from stacks
    if let Some(ref pf) = tree.config.preflop {
        let (sb_seat, bb_seat) = solver::tree::position::blind_seats(num_players);
        stacks[sb_seat] -= pf.blinds[0];
        stacks[bb_seat] -= pf.blinds[1];
    }

    // Navigate through path
    for (i, &action_idx) in indices.iter().enumerate() {
        match current_node {
            ActionTreeNode::Player {
                player,
                actions,
                children,
            } => {
                if action_idx >= actions.len() {
                    return error_state(&format!(
                        "Invalid action index {} at path position {}",
                        action_idx, i
                    ));
                }

                let action = &actions[action_idx];
                let player_name = positions[*player].short_name().to_string();

                // Update state based on action
                update_state_for_action(&mut pot, &mut stacks, *player, action);

                // Record history
                action_history.push(ActionHistoryItem {
                    player: *player,
                    player_name,
                    action: action.display(),
                    path: current_path.clone(),
                });

                // Update path
                if current_path.is_empty() {
                    current_path = action_idx.to_string();
                } else {
                    current_path = format!("{}.{}", current_path, action_idx);
                }

                current_node = &children[action_idx];
            }
            ActionTreeNode::Chance {
                street: next_street,
                child,
            } => {
                street = *next_street;
                current_node = child;
                // Chance nodes don't consume path indices in our navigation
                // but we need to handle them in the display
            }
            ActionTreeNode::Terminal { .. } => {
                return error_state("Cannot navigate past terminal node");
            }
        }
    }

    // Build result based on current node
    build_node_state(
        current_node,
        positions,
        pot,
        stacks,
        street,
        action_history,
        path.to_string(),
    )
}

fn update_state_for_action(pot: &mut i32, stacks: &mut [i32], player: usize, action: &Action) {
    match action {
        Action::Fold | Action::Check => {}
        Action::Call(amount) | Action::Bet(amount) | Action::Raise(amount) | Action::AllIn(amount) => {
            // This is a simplified update - the actual pot tracking would need
            // to account for the betting state more precisely
            let bet_amount = *amount;
            if stacks[player] >= bet_amount {
                stacks[player] -= bet_amount;
                *pot += bet_amount;
            }
        }
    }
}

fn build_node_state(
    node: &ActionTreeNode,
    positions: &[Position],
    pot: i32,
    stacks: Vec<i32>,
    street: Street,
    action_history: Vec<ActionHistoryItem>,
    path: String,
) -> NodeState {
    match node {
        ActionTreeNode::Player {
            player,
            actions,
            ..
        } => {
            let player_name = positions[*player].short_name().to_string();
            let action_infos: Vec<ActionInfo> = actions
                .iter()
                .enumerate()
                .map(|(i, a)| ActionInfo {
                    index: i,
                    name: a.display(),
                    short_name: action_short_name(a),
                    action_type: action_type_name(a),
                    amount: action_amount(a),
                })
                .collect();

            NodeState {
                node_type: "player".to_string(),
                street: street.short_name().to_string(),
                player_to_act: Some(*player),
                player_name: Some(player_name),
                pot,
                stacks,
                actions: action_infos,
                action_history,
                path,
                terminal_result: None,
                error: None,
            }
        }
        ActionTreeNode::Chance {
            street: next_street,
            child,
        } => {
            // For chance nodes, we show the state after the card is dealt
            // and automatically navigate to the child
            build_node_state(
                child,
                positions,
                pot,
                stacks,
                *next_street,
                action_history,
                path,
            )
        }
        ActionTreeNode::Terminal { result, pot: final_pot } => {
            let terminal_str = match result {
                TerminalResult::Fold { winner } => {
                    format!("Fold - {} wins", positions[*winner].short_name())
                }
                TerminalResult::Showdown => "Showdown".to_string(),
                TerminalResult::AllInRunout { num_players } => {
                    format!("All-in Runout ({} players)", num_players)
                }
            };

            NodeState {
                node_type: "terminal".to_string(),
                street: street.short_name().to_string(),
                player_to_act: None,
                player_name: None,
                pot: *final_pot,
                stacks,
                actions: vec![],
                action_history,
                path,
                terminal_result: Some(terminal_str),
                error: None,
            }
        }
    }
}

fn action_short_name(action: &Action) -> String {
    match action {
        Action::Fold => "F".to_string(),
        Action::Check => "X".to_string(),
        Action::Call(a) => format!("C{}", a),
        Action::Bet(a) => format!("B{}", a),
        Action::Raise(a) => format!("R{}", a),
        Action::AllIn(a) => format!("A{}", a),
    }
}

fn action_type_name(action: &Action) -> String {
    match action {
        Action::Fold => "fold".to_string(),
        Action::Check => "check".to_string(),
        Action::Call(_) => "call".to_string(),
        Action::Bet(_) => "bet".to_string(),
        Action::Raise(_) => "raise".to_string(),
        Action::AllIn(_) => "allin".to_string(),
    }
}

fn action_amount(action: &Action) -> Option<i32> {
    match action {
        Action::Fold | Action::Check => None,
        Action::Call(a) | Action::Bet(a) | Action::Raise(a) | Action::AllIn(a) => Some(*a),
    }
}

fn error_state(message: &str) -> NodeState {
    NodeState {
        node_type: "error".to_string(),
        street: "".to_string(),
        player_to_act: None,
        player_name: None,
        pot: 0,
        stacks: vec![],
        actions: vec![],
        action_history: vec![],
        path: "".to_string(),
        terminal_result: None,
        error: Some(message.to_string()),
    }
}
