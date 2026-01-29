// Tauri bridge - provides the same API as the WASM module
// This file wraps Tauri invoke() calls to match the WASM function signatures

// Lazy accessor for Tauri invoke - handles case where module loads before Tauri is ready
function getInvoke() {
    if (window.__TAURI__ && window.__TAURI__.core && window.__TAURI__.core.invoke) {
        return window.__TAURI__.core.invoke;
    }
    throw new Error('Tauri API not available. Make sure withGlobalTauri is enabled.');
}

// === Tree Building ===

export async function build_tree(config_json) {
    return await getInvoke()('build_tree', { configJson: config_json });
}

export async function validate_config(config_json) {
    return await getInvoke()('validate_config', { configJson: config_json });
}

export async function parse_bet_size_preview(size_str, pot, stack) {
    return await getInvoke()('parse_bet_size_preview', { sizeStr: size_str, pot, stack });
}

export async function get_default_config() {
    return await getInvoke()('get_default_config');
}

// === Tree Navigation ===

export async function get_tree_root(config_json) {
    return await getInvoke()('get_tree_root', { configJson: config_json });
}

export async function get_tree_children(config_json, path) {
    return await getInvoke()('get_tree_children', { configJson: config_json, path });
}

export async function get_node_at_path(config_json, path) {
    return await getInvoke()('get_node_at_path', { configJson: config_json, path });
}

export async function get_action_result(config_json, current_path, action_index) {
    return await getInvoke()('get_action_result', { configJson: config_json, path: current_path, actionIdx: action_index });
}

// === Solver ===

export async function create_solver(config_json) {
    return await getInvoke()('create_solver', { configJson: config_json });
}

export async function run_iterations(count) {
    return await getInvoke()('run_iterations', { count });
}

export async function get_exploitability() {
    return await getInvoke()('get_exploitability');
}

// === Strategy ===

export async function get_node_strategy(path, player) {
    return await getInvoke()('get_node_strategy', { path, player });
}

export async function get_node_strategy_for_context(path, player, card_context) {
    return await getInvoke()('get_node_strategy_for_context', { path, player, context: card_context });
}

export async function get_river_cards() {
    return await getInvoke()('get_river_cards');
}

export async function get_turn_cards() {
    return await getInvoke()('get_turn_cards');
}

export async function is_node_below_chance(path) {
    return await getInvoke()('is_node_below_chance', { path });
}

export async function get_chance_depth(path) {
    return await getInvoke()('get_chance_depth', { path });
}

// === Initialization ===

// Tauri doesn't need async init like WASM - provide a no-op for compatibility
export async function init() {
    // Wait a moment for Tauri to be ready, then verify
    return new Promise((resolve, reject) => {
        // Check immediately
        if (window.__TAURI__) {
            resolve();
            return;
        }
        // Otherwise wait a bit and check again
        let attempts = 0;
        const check = () => {
            attempts++;
            if (window.__TAURI__) {
                resolve();
            } else if (attempts < 50) {
                setTimeout(check, 100);
            } else {
                reject(new Error('Tauri API not available after timeout'));
            }
        };
        setTimeout(check, 100);
    });
}

// Default export for compatibility with: import init from '...'
export default init;
