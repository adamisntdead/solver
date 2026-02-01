/* @ts-self-types="./solver_wasm.d.ts" */

/**
 * Build a tree from JSON config and return stats.
 * @param {string} config_json
 * @returns {any}
 */
export function build_tree(config_json) {
    const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.build_tree(ptr0, len0);
    return ret;
}

/**
 * Clear all loaded abstractions.
 */
export function clear_abstractions() {
    wasm.clear_abstractions();
}

/**
 * Compute EHS for all hands on a board.
 *
 * Board should be 3-5 cards (e.g., "KhQsJs2c3d" for river).
 * @param {string} board_str
 * @returns {any}
 */
export function compute_board_ehs(board_str) {
    const ptr0 = passStringToWasm0(board_str, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.compute_board_ehs(ptr0, len0);
    return ret;
}

/**
 * Compute win/split frequencies for all hands on a river board.
 * @param {string} board_str
 * @returns {any}
 */
export function compute_board_winsplit(board_str) {
    const ptr0 = passStringToWasm0(board_str, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.compute_board_winsplit(ptr0, len0);
    return ret;
}

/**
 * Compute the EMD histogram for a specific hand on a non-river board.
 * @param {string} board_str
 * @param {string} combo_str
 * @returns {any}
 */
export function compute_emd_histogram(board_str, combo_str) {
    const ptr0 = passStringToWasm0(board_str, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(combo_str, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.compute_emd_histogram(ptr0, len0, ptr1, len1);
    return ret;
}

/**
 * Create a solver from JSON config.
 *
 * Config includes board, ranges, pot, stack, and tree configuration.
 * @param {string} config_json
 * @returns {any}
 */
export function create_solver(config_json) {
    const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.create_solver(ptr0, len0);
    return ret;
}

/**
 * Get the result of taking an action from the current path.
 * @param {string} config_json
 * @param {string} current_path
 * @param {number} action_index
 * @returns {any}
 */
export function get_action_result(config_json, current_path, action_index) {
    const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(current_path, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.get_action_result(ptr0, len0, ptr1, len1, action_index);
    return ret;
}

/**
 * Get AggSI (Aggressive Suit Isomorphism) buckets for a board.
 * @param {string} board_str
 * @returns {any}
 */
export function get_aggsi_buckets(board_str) {
    const ptr0 = passStringToWasm0(board_str, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.get_aggsi_buckets(ptr0, len0);
    return ret;
}

/**
 * Get the chance depth for a node at the given path.
 *
 * Returns 0 for nodes before any chance node, 1 for nodes after the first
 * chance node (turn betting in flop trees, river betting in turn trees),
 * and 2 for nodes after the second chance node (river betting in flop trees).
 * @param {string} path
 * @returns {number}
 */
export function get_chance_depth(path) {
    const ptr0 = passStringToWasm0(path, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.get_chance_depth(ptr0, len0);
    return ret;
}

/**
 * Get a default config as JSON.
 * @returns {string}
 */
export function get_default_config() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_default_config();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Get current exploitability.
 * @returns {number}
 */
export function get_exploitability() {
    const ret = wasm.get_exploitability();
    return ret;
}

/**
 * Check which abstractions are currently loaded.
 * @returns {any}
 */
export function get_loaded_abstractions() {
    const ret = wasm.get_loaded_abstractions();
    return ret;
}

/**
 * Get the node state at a given path for tree navigation.
 * @param {string} config_json
 * @param {string} path
 * @returns {any}
 */
export function get_node_at_path(config_json, path) {
    const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(path, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.get_node_at_path(ptr0, len0, ptr1, len1);
    return ret;
}

/**
 * Get the strategy for a specific node after solving.
 *
 * `path` is a dot-separated string of action indices (e.g., "0.1.2").
 * `player` is the acting player at this node (0=IP, 1=OOP).
 * @param {string} path
 * @param {number} player
 * @returns {any}
 */
export function get_node_strategy(path, player) {
    const ptr0 = passStringToWasm0(path, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.get_node_strategy(ptr0, len0, player);
    return ret;
}

/**
 * Get the strategy for a specific node with a specific card context.
 *
 * `card_context`: -1 for average across all river cards, 0+ for a specific river card index.
 * When a specific card context is selected, hands that conflict with the river card are excluded.
 * @param {string} path
 * @param {number} player
 * @param {number} card_context
 * @returns {any}
 */
export function get_node_strategy_for_context(path, player, card_context) {
    const ptr0 = passStringToWasm0(path, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.get_node_strategy_for_context(ptr0, len0, player, card_context);
    return ret;
}

/**
 * Get the list of valid river cards for the current solver.
 *
 * Returns { has_river_cards: bool, cards: ["2c", "2d", ...] }.
 * `has_river_cards` is true when solving a turn spot (4-card board).
 * @returns {any}
 */
export function get_river_cards() {
    const ret = wasm.get_river_cards();
    return ret;
}

/**
 * Get children of a node at the given path for lazy loading.
 * @param {string} config_json
 * @param {string} path
 * @returns {any}
 */
export function get_tree_children(config_json, path) {
    const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(path, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.get_tree_children(ptr0, len0, ptr1, len1);
    return ret;
}

/**
 * Get the tree root with its immediate children for the tree view.
 * @param {string} config_json
 * @returns {any}
 */
export function get_tree_root(config_json) {
    const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.get_tree_root(ptr0, len0);
    return ret;
}

/**
 * Get the list of valid turn cards for the current solver.
 *
 * Returns { has_turn_cards: bool, cards: ["2c", "2d", ...] }.
 * `has_turn_cards` is true when solving a flop spot (3-card board).
 * @returns {any}
 */
export function get_turn_cards() {
    const ret = wasm.get_turn_cards();
    return ret;
}

/**
 * Initialize panic hook for better error messages in browser console.
 */
export function init() {
    wasm.init();
}

/**
 * Check if a node at the given path is below a chance node (has multiple card contexts).
 *
 * Returns true if the node has more than 1 card context (i.e., it's a river betting node
 * in a turn tree).
 * @param {string} path
 * @returns {boolean}
 */
export function is_node_below_chance(path) {
    const ptr0 = passStringToWasm0(path, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.is_node_below_chance(ptr0, len0);
    return ret !== 0;
}

/**
 * Load an abstraction file for a specific street.
 *
 * The `street` parameter should be "flop", "turn", or "river".
 * The `data` parameter is the raw bytes of the .abs file.
 *
 * Call this before create_solver() to enable hand abstraction.
 * @param {string} street
 * @param {Uint8Array} data
 * @returns {any}
 */
export function load_abstraction(street, data) {
    const ptr0 = passStringToWasm0(street, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(data, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.load_abstraction(ptr0, len0, ptr1, len1);
    return ret;
}

/**
 * Look up a specific hand's information on a board.
 * @param {string} board_str
 * @param {string} combo_str
 * @returns {any}
 */
export function lookup_hand(board_str, combo_str) {
    const ptr0 = passStringToWasm0(board_str, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(combo_str, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.lookup_hand(ptr0, len0, ptr1, len1);
    return ret;
}

/**
 * Preview a bet size resolution.
 * @param {string} size_str
 * @param {number} pot
 * @param {number} stack
 * @returns {any}
 */
export function parse_bet_size_preview(size_str, pot, stack) {
    const ptr0 = passStringToWasm0(size_str, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.parse_bet_size_preview(ptr0, len0, pot, stack);
    return ret;
}

/**
 * Run solver iterations.
 *
 * Returns total iteration count and current exploitability.
 * @param {number} count
 * @returns {any}
 */
export function run_iterations(count) {
    const ret = wasm.run_iterations(count);
    return ret;
}

/**
 * Validate a config without building the full tree.
 * @param {string} config_json
 * @returns {any}
 */
export function validate_config(config_json) {
    const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.validate_config(ptr0, len0);
    return ret;
}

function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Error_8c4e43fe74559d73: function(arg0, arg1) {
            const ret = Error(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg___wbindgen_debug_string_0bc8482c6e3508ae: function(arg0, arg1) {
            const ret = debugString(arg1);
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_throw_be289d5034ed271b: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_error_7534b8e9a36f1ab4: function(arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
            }
        },
        __wbg_new_361308b2356cecd0: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_3eb36ae241fe6f44: function() {
            const ret = new Array();
            return ret;
        },
        __wbg_new_8a6f238a6ece86ea: function() {
            const ret = new Error();
            return ret;
        },
        __wbg_set_3f1d0b984ed272ed: function(arg0, arg1, arg2) {
            arg0[arg1] = arg2;
        },
        __wbg_set_f43e577aea94465b: function(arg0, arg1, arg2) {
            arg0[arg1 >>> 0] = arg2;
        },
        __wbg_stack_0ed75d68575b0f3c: function(arg0, arg1) {
            const ret = arg1.stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbindgen_cast_0000000000000001: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return ret;
        },
        __wbindgen_cast_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_cast_0000000000000003: function(arg0) {
            // Cast intrinsic for `U64 -> Externref`.
            const ret = BigInt.asUintN(64, arg0);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./solver_wasm_bg.js": import0,
    };
}

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasm;
function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('solver_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
