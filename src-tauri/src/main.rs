// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod state;

use state::SolverState;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(SolverState::new())
        .invoke_handler(tauri::generate_handler![
            // Tree building
            commands::build_tree,
            commands::validate_config,
            commands::parse_bet_size_preview,
            commands::get_default_config,
            // Tree navigation
            commands::get_tree_root,
            commands::get_tree_children,
            commands::get_node_at_path,
            commands::get_action_result,
            // Solver
            commands::create_solver,
            commands::run_iterations,
            commands::get_exploitability,
            // Strategy
            commands::get_node_strategy,
            commands::get_node_strategy_for_context,
            commands::get_river_cards,
            commands::get_turn_cards,
            commands::is_node_below_chance,
            commands::get_chance_depth,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
