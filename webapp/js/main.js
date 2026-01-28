// Main entry point for the Poker Tree Builder webapp

import init, {
    build_tree,
    validate_config,
    parse_bet_size_preview,
    get_node_at_path,
    get_action_result,
    get_default_config,
    get_tree_root,
    get_tree_children,
    create_solver,
    run_iterations,
    get_node_strategy,
    get_exploitability,
    get_river_cards,
    is_node_below_chance,
    get_node_strategy_for_context
} from '../pkg/solver_wasm.js';

import * as TreeView from './tree-view.js';

// === State ===
let wasmLoaded = false;
let currentConfig = null;
let currentPath = '';
let treeBuilt = false;
let currentNodeState = null;

// Solver state
let solverActive = false;
let solverRunning = false;
let solverStopped = false;
let solverTotalIterations = 0;

// River card selector state
let selectedRiverCardContext = -1; // -1 = average, 0+ = specific card index
let riverCardInfo = null;          // result from get_river_cards()

// === DOM Elements ===
const el = {
    // Config
    spotName: document.getElementById('spot-name'),
    numPlayers: document.getElementById('num-players'),
    startingStreet: document.getElementById('starting-street'),
    startingPot: document.getElementById('starting-pot'),
    postflopOptions: document.getElementById('postflop-options'),
    preflopOptions: document.getElementById('preflop-options'),
    stacksContainer: document.getElementById('stacks-container'),
    perPlayerStacks: document.getElementById('per-player-stacks'),
    sbSize: document.getElementById('sb-size'),
    bbSize: document.getElementById('bb-size'),
    betType: document.getElementById('bet-type'),
    maxRaises: document.getElementById('max-raises'),
    preflopOpenRaise: document.getElementById('preflop-open-raise'),
    preflop3betRaise: document.getElementById('preflop-3bet-raise'),
    postflopBet: document.getElementById('postflop-bet'),
    postflopRaise: document.getElementById('postflop-raise'),

    // Buttons
    buildBtn: document.getElementById('build-btn'),
    saveBtn: document.getElementById('save-btn'),
    loadBtn: document.getElementById('load-btn'),
    loadInput: document.getElementById('load-input'),
    expandAllBtn: document.getElementById('expand-all-btn'),
    collapseAllBtn: document.getElementById('collapse-all-btn'),

    // Tree
    treeView: document.getElementById('tree-view'),
    breadcrumb: document.getElementById('breadcrumb'),

    // Node Inspector
    nodeDetails: document.getElementById('node-details'),
    streetBadge: document.getElementById('street-badge'),
    playerBadge: document.getElementById('player-badge'),
    potValue: document.getElementById('pot-value'),
    toCallValue: document.getElementById('to-call-value'),
    stackValue: document.getElementById('stack-value'),
    terminalInfo: document.getElementById('terminal-info'),
    terminalText: document.getElementById('terminal-text'),

    // Enabled Actions
    enabledActions: document.getElementById('enabled-actions'),
    actionCount: document.getElementById('action-count'),
    childrenSection: document.getElementById('children-section'),
    paletteSection: document.getElementById('palette-section'),

    // Palette
    addActionsBtn: document.getElementById('add-actions-btn'),
    removeActionsBtn: document.getElementById('remove-actions-btn'),
    customSize: document.getElementById('custom-size'),
    addCustomBtn: document.getElementById('add-custom-btn'),
    applyTemplateBtn: document.getElementById('apply-template-btn'),
    pruneSubtreeBtn: document.getElementById('prune-subtree-btn'),

    // Stats
    statNodes: document.getElementById('stat-nodes'),
    statTerminals: document.getElementById('stat-terminals'),
    statDepth: document.getElementById('stat-depth'),
    statMemory: document.getElementById('stat-memory'),

    // Status
    statusBar: document.getElementById('status-bar'),

    // Solver
    solverSection: document.getElementById('solver-section'),
    solverUnsupported: document.getElementById('solver-unsupported'),
    solverBoardGroup: document.getElementById('solver-board-group'),
    solverBoard: document.getElementById('solver-board'),
    solverOopRange: document.getElementById('solver-oop-range'),
    solverIpRange: document.getElementById('solver-ip-range'),
    solverIterations: document.getElementById('solver-iterations'),
    solveBtn: document.getElementById('solve-btn'),
    stopSolveBtn: document.getElementById('stop-solve-btn'),
    solverProgress: document.getElementById('solver-progress'),
    progressFill: document.getElementById('progress-fill'),
    solverIterCount: document.getElementById('solver-iter-count'),
    solverExploit: document.getElementById('solver-exploit'),

    // Strategy
    strategySection: document.getElementById('strategy-section'),
    strategyAggregate: document.getElementById('strategy-aggregate'),
    strategyLegend: document.getElementById('strategy-legend'),
    strategyThead: document.getElementById('strategy-thead'),
    strategyTbody: document.getElementById('strategy-tbody'),

    // River card selector
    riverCardSelector: document.getElementById('river-card-selector'),
    riverCardGrid: document.getElementById('river-card-grid')
};

// === Initialize ===
async function initialize() {
    setStatus('Loading WASM module...');

    try {
        await init();
        wasmLoaded = true;
        setStatus('Ready - Press Ctrl+B to build tree', 'success');

        // Initialize tree view
        TreeView.initTreeView(el.treeView, handleNodeSelect);

        // Expose toggle function for tree view
        window.treeViewToggle = (path) => {
            TreeView.toggleNode(path, getTreeChildren);
        };

        // Load default config
        const defaultConfig = get_default_config();
        currentConfig = JSON.parse(defaultConfig);
        loadConfigToUI(currentConfig);

        // Set up event listeners
        setupEventListeners();

    } catch (error) {
        setStatus(`Failed to load WASM: ${error.message}`, 'error');
        console.error('WASM init error:', error);
    }
}

// === Event Listeners ===
function setupEventListeners() {
    // Build
    el.buildBtn.addEventListener('click', buildTree);

    // Save/Load
    el.saveBtn.addEventListener('click', saveConfig);
    el.loadBtn.addEventListener('click', () => el.loadInput.click());
    el.loadInput.addEventListener('change', loadConfig);

    // Street selection
    el.startingStreet.addEventListener('change', handleStreetChange);

    // Player count change
    el.numPlayers.addEventListener('change', handlePlayerCountChange);

    // Per-player stacks toggle
    el.perPlayerStacks.addEventListener('change', handleStackModeChange);

    // Tree controls
    el.expandAllBtn.addEventListener('click', handleExpandAll);
    el.collapseAllBtn.addEventListener('click', handleCollapseAll);

    // Palette buttons
    document.querySelectorAll('.palette-btn[data-action]').forEach(btn => {
        btn.addEventListener('click', () => togglePaletteButton(btn));
    });

    el.addActionsBtn.addEventListener('click', handleAddActions);
    el.removeActionsBtn.addEventListener('click', handleRemoveActions);
    el.addCustomBtn.addEventListener('click', handleAddCustom);
    el.applyTemplateBtn.addEventListener('click', handleApplyTemplate);
    el.pruneSubtreeBtn.addEventListener('click', handlePruneSubtree);

    // Solver
    el.solveBtn.addEventListener('click', startSolve);
    el.stopSolveBtn.addEventListener('click', stopSolve);

    // Keyboard shortcuts
    document.addEventListener('keydown', handleGlobalKeyDown);
}

// === Street Selection ===
function handleStreetChange() {
    const street = el.startingStreet.value;
    const isPostflop = street !== 'preflop';

    // Show/hide postflop options (starting pot)
    el.postflopOptions.style.display = isPostflop ? 'flex' : 'none';

    // Show/hide preflop options (blinds)
    el.preflopOptions.style.display = isPostflop ? 'none' : 'flex';

    // Solver only supports turn and river
    updateSolverAvailability(street);
}

function updateSolverAvailability(street) {
    const canSolve = street === 'turn' || street === 'river';
    el.solverUnsupported.style.display = canSolve ? 'none' : 'block';
    el.solverBoardGroup.style.display = canSolve ? '' : 'none';
    el.solveBtn.disabled = !canSolve || !treeBuilt;

    // Update board placeholder for the street
    if (street === 'turn') {
        el.solverBoard.placeholder = 'e.g. KhQsJs2c';
    } else {
        el.solverBoard.placeholder = 'e.g. KhQsJs2c3d';
    }
}

// === Stack Management ===
function handlePlayerCountChange() {
    const numPlayers = parseInt(el.numPlayers.value);
    if (el.perPlayerStacks.checked) {
        rebuildStackInputs(numPlayers);
    }
}

function handleStackModeChange() {
    const numPlayers = parseInt(el.numPlayers.value);
    if (el.perPlayerStacks.checked) {
        rebuildStackInputs(numPlayers);
    } else {
        rebuildStackInputs(1); // Single "All" input
    }
}

function rebuildStackInputs(count) {
    const currentStack = getFirstStackValue();
    const positions = getPositionNames(parseInt(el.numPlayers.value));

    el.stacksContainer.innerHTML = '';

    if (count === 1) {
        // Single input for all players
        const row = document.createElement('div');
        row.className = 'stack-input-row';
        row.innerHTML = `
            <span class="player-label">All</span>
            <input type="number" class="stack-input" data-player="all" value="${currentStack}" min="1">
        `;
        el.stacksContainer.appendChild(row);
    } else {
        // Per-player inputs
        for (let i = 0; i < count; i++) {
            const row = document.createElement('div');
            row.className = 'stack-input-row';
            row.innerHTML = `
                <span class="player-label">${positions[i] || `P${i+1}`}</span>
                <input type="number" class="stack-input" data-player="${i}" value="${currentStack}" min="1">
            `;
            el.stacksContainer.appendChild(row);
        }
    }
}

function getFirstStackValue() {
    const input = el.stacksContainer.querySelector('.stack-input');
    return parseInt(input?.value) || 200;
}

function getPositionNames(numPlayers) {
    if (numPlayers === 2) return ['BTN', 'BB'];
    if (numPlayers === 3) return ['BTN', 'SB', 'BB'];
    if (numPlayers === 4) return ['CO', 'BTN', 'SB', 'BB'];
    if (numPlayers === 5) return ['MP', 'CO', 'BTN', 'SB', 'BB'];
    if (numPlayers === 6) return ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB'];
    return Array.from({ length: numPlayers }, (_, i) => `P${i+1}`);
}

function getStacksFromUI() {
    const inputs = el.stacksContainer.querySelectorAll('.stack-input');
    const stacks = [];
    inputs.forEach(input => {
        stacks.push(parseInt(input.value) || 100);
    });
    return stacks;
}

function handleGlobalKeyDown(e) {
    // Ctrl+B to build
    if (e.ctrlKey && e.key === 'b') {
        e.preventDefault();
        buildTree();
    }
    // Ctrl+E to expand all
    if (e.ctrlKey && e.key === 'e') {
        e.preventDefault();
        handleExpandAll();
    }
    // Ctrl+W to collapse all
    if (e.ctrlKey && e.key === 'w') {
        e.preventDefault();
        handleCollapseAll();
    }
}

// === Build Tree ===
async function buildTree() {
    if (!wasmLoaded) {
        setStatus('WASM not loaded', 'error');
        return;
    }

    setStatus('Building tree...');

    try {
        currentConfig = collectConfigFromUI();
        const configJson = JSON.stringify(currentConfig);

        // Validate
        const validation = validate_config(configJson);
        if (!validation.valid) {
            setStatus(`Validation errors: ${validation.errors.join(', ')}`, 'error');
            return;
        }

        // Build
        const result = build_tree(configJson);
        if (!result.success) {
            setStatus(`Build failed: ${result.error}`, 'error');
            return;
        }

        // Update stats
        updateStats(result.stats, result.memory);

        treeBuilt = true;
        currentPath = '';
        solverActive = false;
        selectedRiverCardContext = -1;
        riverCardInfo = null;
        el.strategySection.style.display = 'none';
        updateSolverAvailability(el.startingStreet.value);

        // Load tree view
        await loadTreeView();

        // Select root
        handleNodeSelect('');

        setStatus('Tree built successfully', 'success');

    } catch (error) {
        setStatus(`Build error: ${error.message}`, 'error');
        console.error('Build error:', error);
    }
}

// === Config ===
function collectConfigFromUI() {
    const sb = parseInt(el.sbSize.value) || 1;
    const bb = parseInt(el.bbSize.value) || 2;
    const startingStreet = el.startingStreet.value;
    const isPreflop = startingStreet === 'preflop';

    // Get stacks directly in chips (no BB conversion)
    const stacks = getStacksFromUI();

    const config = {
        name: el.spotName.value,
        num_players: parseInt(el.numPlayers.value),
        starting_stacks: stacks,
        starting_street: startingStreet,
        starting_pot: isPreflop ? 0 : parseInt(el.startingPot.value) || 100,
        bet_type: el.betType.value,
        max_raises_per_round: parseInt(el.maxRaises.value) || 4,
        force_all_in_threshold: 0.15,
        merge_threshold: 0.1,
        add_all_in_threshold: 1.5
    };

    // Only include preflop config if starting from preflop
    if (isPreflop) {
        config.preflop = {
            blinds: [sb, bb],
            ante: 0,
            bb_ante: 0,
            open_sizes: {
                bet: '',
                raise: el.preflopOpenRaise.value
            },
            three_bet_sizes: {
                bet: '',
                raise: el.preflop3betRaise.value
            },
            four_bet_sizes: {
                bet: '',
                raise: '2.2x, a'
            },
            allow_limps: true
        };
    }

    // Include postflop configs based on starting street
    const postflopConfig = {
        sizes: {
            bet: el.postflopBet.value,
            raise: el.postflopRaise.value
        }
    };

    // Include streets from starting street onwards
    const streets = ['preflop', 'flop', 'turn', 'river'];
    const startIdx = streets.indexOf(startingStreet);

    if (startIdx <= 1) config.flop = postflopConfig;
    if (startIdx <= 2) config.turn = { ...postflopConfig };
    if (startIdx <= 3) config.river = { ...postflopConfig };

    return config;
}

function loadConfigToUI(config) {
    el.spotName.value = config.name || 'HU 100bb';
    el.numPlayers.value = config.num_players || 2;

    // Starting street
    const startingStreet = config.starting_street || 'preflop';
    el.startingStreet.value = startingStreet;
    handleStreetChange(); // Update UI visibility

    // Starting pot (for postflop)
    el.startingPot.value = config.starting_pot || 100;

    // Blinds
    el.sbSize.value = config.preflop?.blinds?.[0] || 1;
    el.bbSize.value = config.preflop?.blinds?.[1] || 2;

    // Stacks (directly in chips, no BB conversion)
    const stacks = config.starting_stacks || [200];

    if (stacks.length > 1) {
        // Per-player stacks
        el.perPlayerStacks.checked = true;
        rebuildStackInputs(stacks.length);
        const inputs = el.stacksContainer.querySelectorAll('.stack-input');
        inputs.forEach((input, i) => {
            input.value = stacks[i] || stacks[0];
        });
    } else {
        // Single stack for all
        el.perPlayerStacks.checked = false;
        rebuildStackInputs(1);
        const input = el.stacksContainer.querySelector('.stack-input');
        if (input) input.value = stacks[0] || 200;
    }

    el.betType.value = config.bet_type || 'NoLimit';
    el.maxRaises.value = config.max_raises_per_round || 4;

    el.preflopOpenRaise.value = config.preflop?.open_sizes?.raise || '2.5x, 3x';
    el.preflop3betRaise.value = config.preflop?.three_bet_sizes?.raise || '3x, a';
    el.postflopBet.value = config.flop?.sizes?.bet || '33%, 67%, 100%';
    el.postflopRaise.value = config.flop?.sizes?.raise || '2.5x, a';
}

function saveConfig() {
    const config = collectConfigFromUI();
    const json = JSON.stringify(config, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `${config.name.replace(/\s+/g, '_')}.json`;
    a.click();

    URL.revokeObjectURL(url);
    setStatus('Config saved', 'success');
}

function loadConfig(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const config = JSON.parse(e.target.result);
            currentConfig = config;
            loadConfigToUI(config);
            setStatus('Config loaded - click Build to apply', 'success');
        } catch (error) {
            setStatus(`Failed to load config: ${error.message}`, 'error');
        }
    };
    reader.readAsText(file);
    event.target.value = '';
}

// === Tree View ===
async function loadTreeView() {
    if (!wasmLoaded || !currentConfig) return;

    try {
        const configJson = JSON.stringify(currentConfig);
        const root = get_tree_root(configJson);
        if (root) {
            TreeView.setTreeData(root);
        }
    } catch (error) {
        console.error('Error loading tree view:', error);
        TreeView.clearTree();
    }
}

async function getTreeChildren(path) {
    if (!wasmLoaded || !currentConfig) return [];

    try {
        const configJson = JSON.stringify(currentConfig);
        return get_tree_children(configJson, path) || [];
    } catch (error) {
        console.error('Error getting children:', error);
        return [];
    }
}

function handleNodeSelect(path) {
    currentPath = path;
    TreeView.setSelectedPath(path);
    navigateToPath(path);
}

async function handleExpandAll() {
    if (!treeBuilt) return;
    setStatus('Expanding tree...');
    await TreeView.expandAll(getTreeChildren, 3);
    setStatus('Tree expanded', 'success');
}

function handleCollapseAll() {
    TreeView.collapseAll();
}

// === Navigation ===
function navigateToPath(path) {
    if (!wasmLoaded || !treeBuilt) return;

    try {
        const configJson = JSON.stringify(currentConfig);
        const nodeState = get_node_at_path(configJson, path);

        if (nodeState.error) {
            setStatus(`Navigation error: ${nodeState.error}`, 'error');
            return;
        }

        currentNodeState = nodeState;
        updateNodeInspector(nodeState);
        updateBreadcrumb(nodeState);

    } catch (error) {
        setStatus(`Navigation error: ${error.message}`, 'error');
        console.error('Navigation error:', error);
    }
}

// === Node Inspector ===
function updateNodeInspector(state) {
    // Street badge
    const street = state.street?.toLowerCase() || 'preflop';
    el.streetBadge.textContent = state.street || 'PREFLOP';
    el.streetBadge.className = `street-badge ${street}`;

    if (state.node_type === 'terminal') {
        // Terminal node
        el.nodeDetails.style.display = 'none';
        el.terminalInfo.style.display = 'flex';
        el.terminalText.textContent = state.terminal_result || 'Terminal';
        el.childrenSection.style.display = 'none';
        el.paletteSection.style.display = 'none';
        el.strategySection.style.display = 'none';
    } else {
        // Player node
        el.nodeDetails.style.display = 'block';
        el.terminalInfo.style.display = 'none';
        el.childrenSection.style.display = 'block';
        el.paletteSection.style.display = 'block';

        el.playerBadge.textContent = state.player_name || 'P' + state.player_to_act;
        el.potValue.textContent = state.pot || 0;
        el.toCallValue.textContent = getToCall(state);
        el.stackValue.textContent = state.stacks?.[state.player_to_act] || 0;

        // Update enabled actions (children)
        updateEnabledActions(state.actions || []);

        // Show strategy if solver is active
        if (solverActive && state.player_to_act !== undefined) {
            displayNodeStrategy(currentPath, state.player_to_act);
        } else {
            el.strategySection.style.display = 'none';
        }
    }
}

function getToCall(state) {
    // Calculate amount to call from last action
    // This is simplified - would need more state tracking for accuracy
    return 0;
}

function updateEnabledActions(actions) {
    el.actionCount.textContent = `(${actions.length})`;

    if (actions.length === 0) {
        el.enabledActions.innerHTML = '<div class="empty-state">No actions at this node</div>';
        return;
    }

    el.enabledActions.innerHTML = '';

    for (const action of actions) {
        const item = document.createElement('div');
        item.className = 'enabled-action';
        item.dataset.index = action.index;

        const icon = document.createElement('span');
        icon.className = `enabled-action-icon ${action.action_type}`;
        item.appendChild(icon);

        const label = document.createElement('span');
        label.className = 'enabled-action-label';
        label.textContent = action.name;
        item.appendChild(label);

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'enabled-action-delete';
        deleteBtn.innerHTML = '&times;';
        deleteBtn.title = 'Remove action';
        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            // TODO: Implement action removal
            setStatus('Action removal not yet implemented', 'warning');
        });
        item.appendChild(deleteBtn);

        // Click to navigate to child
        item.addEventListener('click', () => {
            handleActionClick(action.index);
        });

        el.enabledActions.appendChild(item);
    }
}

async function handleActionClick(actionIndex) {
    if (!wasmLoaded || !treeBuilt) return;

    try {
        const configJson = JSON.stringify(currentConfig);
        const nodeState = get_action_result(configJson, currentPath, actionIndex);

        if (nodeState.error) {
            setStatus(`Action error: ${nodeState.error}`, 'error');
            return;
        }

        currentPath = nodeState.path;

        // Expand path and select in tree
        await TreeView.expandToPath(currentPath, getTreeChildren);
        TreeView.setSelectedPath(currentPath);

        currentNodeState = nodeState;
        updateNodeInspector(nodeState);
        updateBreadcrumb(nodeState);

    } catch (error) {
        setStatus(`Action error: ${error.message}`, 'error');
    }
}

// === Breadcrumb ===
function updateBreadcrumb(state) {
    el.breadcrumb.innerHTML = '';

    // Root
    const rootCrumb = document.createElement('span');
    rootCrumb.className = `crumb ${currentPath === '' ? 'active' : ''}`;
    rootCrumb.textContent = 'Root';
    rootCrumb.addEventListener('click', () => handleNodeSelect(''));
    el.breadcrumb.appendChild(rootCrumb);

    // Action history
    const history = state.action_history || [];
    for (let i = 0; i < history.length; i++) {
        const item = history[i];

        const sep = document.createElement('span');
        sep.className = 'crumb-sep';
        sep.textContent = '>';
        el.breadcrumb.appendChild(sep);

        const crumb = document.createElement('span');
        crumb.className = `crumb ${i === history.length - 1 ? 'active' : ''}`;
        crumb.textContent = `${item.player_name} ${item.action}`;

        const pathParts = currentPath.split('.');
        const pathToHere = pathParts.slice(0, i + 1).join('.');
        crumb.addEventListener('click', () => handleNodeSelect(pathToHere));

        el.breadcrumb.appendChild(crumb);
    }
}

// === Action Palette ===
function togglePaletteButton(btn) {
    btn.classList.toggle('selected');
}

function getSelectedPaletteActions() {
    const selected = [];
    document.querySelectorAll('.palette-btn.selected[data-action]').forEach(btn => {
        selected.push(btn.dataset.action);
    });
    return selected;
}

function handleAddActions() {
    const selected = getSelectedPaletteActions();
    if (selected.length === 0) {
        setStatus('Select actions from the palette first', 'warning');
        return;
    }
    // TODO: Implement action addition
    setStatus(`Adding actions: ${selected.join(', ')} (not yet implemented)`, 'warning');
}

function handleRemoveActions() {
    // TODO: Implement action removal
    setStatus('Action removal not yet implemented', 'warning');
}

function handleAddCustom() {
    const sizeStr = el.customSize.value.trim();
    if (!sizeStr) {
        setStatus('Enter a custom size', 'warning');
        return;
    }
    // TODO: Implement custom action addition
    setStatus(`Adding custom size: ${sizeStr} (not yet implemented)`, 'warning');
}

function handleApplyTemplate() {
    // TODO: Implement template application
    setStatus('Template application not yet implemented', 'warning');
}

function handlePruneSubtree() {
    // TODO: Implement subtree pruning
    setStatus('Subtree pruning not yet implemented', 'warning');
}

// === Stats ===
function updateStats(stats, memory) {
    if (stats) {
        el.statNodes.textContent = formatNumber(stats.node_count);
        el.statTerminals.textContent = formatNumber(stats.terminal_count);
        el.statDepth.textContent = stats.max_depth;
    }

    if (memory) {
        el.statMemory.textContent = formatBytes(memory.compressed_bytes);
    }
}

// === Utilities ===
function formatNumber(n) {
    if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
    if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
    return n.toLocaleString();
}

function formatBytes(bytes) {
    if (typeof bytes !== 'number') return '-';
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let i = 0;
    while (size >= 1024 && i < units.length - 1) {
        size /= 1024;
        i++;
    }
    return `${size.toFixed(1)} ${units[i]}`;
}

function setStatus(message, type = '') {
    el.statusBar.textContent = message;
    el.statusBar.className = `status-bar ${type}`;
}

// === Solver ===
const ACTION_COLORS = [
    '#6b7280', // fold - gray
    '#3b82f6', // check/call - blue
    '#22c55e', // bet - green
    '#f59e0b', // raise - yellow
    '#ef4444', // allin - red
    '#8b5cf6', // purple
    '#ec4899', // pink
    '#14b8a6', // teal
];

function getActionColor(name, index) {
    const n = name.toLowerCase();
    if (n === 'fold') return ACTION_COLORS[0];
    if (n === 'check' || n === 'call') return ACTION_COLORS[1];
    if (n.startsWith('bet')) return ACTION_COLORS[2];
    if (n.startsWith('raise')) return ACTION_COLORS[3];
    if (n.includes('all')) return ACTION_COLORS[4];
    return ACTION_COLORS[Math.min(index + 2, ACTION_COLORS.length - 1)];
}

async function startSolve() {
    if (!wasmLoaded || !treeBuilt || solverRunning) return;

    const board = el.solverBoard.value.trim();
    const oopRange = el.solverOopRange.value.trim();
    const ipRange = el.solverIpRange.value.trim();
    const targetIterations = parseInt(el.solverIterations.value) || 200;

    if (!board) {
        setStatus('Enter a board (e.g., KhQsJs2c3d)', 'error');
        return;
    }
    if (!oopRange) {
        setStatus('Enter an OOP range', 'error');
        return;
    }
    if (!ipRange) {
        setStatus('Enter an IP range', 'error');
        return;
    }

    // Derive pot and effective_stack from tree config
    const config = collectConfigFromUI();
    const pot = config.starting_pot || (config.preflop ? config.preflop.blinds[0] + config.preflop.blinds[1] : 100);
    const effectiveStack = config.starting_stacks[0] || 100;

    // Build solver config
    const solverConfig = {
        board,
        oop_range: oopRange,
        ip_range: ipRange,
        pot,
        effective_stack: effectiveStack,
        tree_config: config
    };

    setStatus('Creating solver...');
    el.solveBtn.style.display = 'none';
    el.stopSolveBtn.style.display = 'block';
    el.solverProgress.style.display = 'block';
    el.progressFill.style.width = '0%';
    el.solverIterCount.textContent = `0 / ${targetIterations}`;
    el.solverExploit.textContent = '-';

    try {
        const createResult = create_solver(JSON.stringify(solverConfig));
        if (!createResult.success) {
            setStatus(`Solver error: ${createResult.error}`, 'error');
            resetSolverUI();
            return;
        }

        solverActive = true;
        solverRunning = true;
        solverStopped = false;
        solverTotalIterations = 0;
        selectedRiverCardContext = -1;

        // Fetch river card info for turn trees
        fetchRiverCards();

        setStatus(`Solver created: ${createResult.num_oop_hands} OOP hands, ${createResult.num_ip_hands} IP hands`);

        // Run iterations in batches
        const batchSize = 50;
        const runBatch = () => {
            if (solverStopped || solverTotalIterations >= targetIterations) {
                // Done
                solverRunning = false;
                el.solveBtn.style.display = 'block';
                el.stopSolveBtn.style.display = 'none';
                el.solveBtn.textContent = 'Re-solve';

                const finalExploit = get_exploitability();
                const exploitPct = pot > 0 ? (finalExploit / pot * 100).toFixed(2) : '?';
                setStatus(`Solve complete: ${solverTotalIterations} iterations, ${exploitPct}% pot exploitability`, 'success');

                // Refresh strategy display for current node
                if (currentNodeState && currentNodeState.node_type !== 'terminal') {
                    displayNodeStrategy(currentPath, currentNodeState.player_to_act);
                }
                return;
            }

            const remaining = targetIterations - solverTotalIterations;
            const count = Math.min(batchSize, remaining);

            const result = run_iterations(count);
            solverTotalIterations = result.total_iterations;

            // Update progress
            const pct = Math.min(100, (solverTotalIterations / targetIterations) * 100);
            el.progressFill.style.width = `${pct}%`;
            el.solverIterCount.textContent = `${solverTotalIterations} / ${targetIterations}`;
            el.solverExploit.textContent = `${result.exploitability_pct.toFixed(2)}% pot`;

            // Yield to UI then run next batch
            setTimeout(runBatch, 0);
        };

        // Start first batch
        setTimeout(runBatch, 0);

    } catch (error) {
        setStatus(`Solver error: ${error.message}`, 'error');
        console.error('Solver error:', error);
        resetSolverUI();
    }
}

function stopSolve() {
    solverStopped = true;
}

function resetSolverUI() {
    solverRunning = false;
    el.solveBtn.style.display = 'block';
    el.stopSolveBtn.style.display = 'none';
}

function fetchRiverCards() {
    try {
        riverCardInfo = get_river_cards();
    } catch (e) {
        riverCardInfo = null;
    }
}

function getSuitClass(card) {
    if (card.length < 2) return '';
    const suit = card[card.length - 1];
    switch (suit) {
        case 'h': return 'suit-h';
        case 'd': return 'suit-d';
        case 'c': return 'suit-c';
        case 's': return 'suit-s';
        default: return '';
    }
}

function renderRiverCardSelector() {
    if (!riverCardInfo || !riverCardInfo.has_river_cards) {
        el.riverCardSelector.style.display = 'none';
        return;
    }

    el.riverCardSelector.style.display = 'block';
    el.riverCardGrid.innerHTML = '';

    // "Avg" button
    const avgBtn = document.createElement('button');
    avgBtn.className = 'river-card-btn' + (selectedRiverCardContext === -1 ? ' selected' : '');
    avgBtn.textContent = 'Avg';
    avgBtn.addEventListener('click', () => {
        selectedRiverCardContext = -1;
        renderRiverCardSelector();
        if (currentNodeState && currentNodeState.node_type !== 'terminal') {
            displayNodeStrategy(currentPath, currentNodeState.player_to_act);
        }
    });
    el.riverCardGrid.appendChild(avgBtn);

    // Per-card buttons
    for (let i = 0; i < riverCardInfo.cards.length; i++) {
        const card = riverCardInfo.cards[i];
        const btn = document.createElement('button');
        const suitCls = getSuitClass(card);
        btn.className = 'river-card-btn ' + suitCls + (selectedRiverCardContext === i ? ' selected' : '');
        btn.textContent = card;
        btn.addEventListener('click', () => {
            selectedRiverCardContext = i;
            renderRiverCardSelector();
            if (currentNodeState && currentNodeState.node_type !== 'terminal') {
                displayNodeStrategy(currentPath, currentNodeState.player_to_act);
            }
        });
        el.riverCardGrid.appendChild(btn);
    }
}

function displayNodeStrategy(path, player) {
    if (!solverActive) {
        el.strategySection.style.display = 'none';
        return;
    }

    try {
        // Check if this node is below a chance node (river betting in turn tree)
        const belowChance = is_node_below_chance(path);

        // Show/hide river card selector
        if (belowChance && riverCardInfo && riverCardInfo.has_river_cards) {
            renderRiverCardSelector();
        } else {
            el.riverCardSelector.style.display = 'none';
            // Reset to avg when navigating to turn-level nodes
            if (!belowChance) {
                selectedRiverCardContext = -1;
            }
        }

        // Use context-aware function when river cards exist
        const result = (riverCardInfo && riverCardInfo.has_river_cards)
            ? get_node_strategy_for_context(path, player, belowChance ? selectedRiverCardContext : -1)
            : get_node_strategy(path, player);

        if (!result.success || result.action_names.length === 0) {
            el.strategySection.style.display = 'none';
            return;
        }

        el.strategySection.style.display = 'block';

        const { action_names, hands, aggregate } = result;

        // Build color map
        const colors = action_names.map((name, i) => getActionColor(name, i));

        // Render aggregate bar
        el.strategyAggregate.innerHTML = '';
        for (let i = 0; i < action_names.length; i++) {
            const pct = (aggregate[i] * 100);
            if (pct < 0.1) continue;
            const seg = document.createElement('div');
            seg.className = 'strategy-segment';
            seg.style.width = `${pct}%`;
            seg.style.background = colors[i];
            seg.title = `${action_names[i]}: ${pct.toFixed(1)}%`;
            el.strategyAggregate.appendChild(seg);
        }

        // Render legend
        el.strategyLegend.innerHTML = '';
        for (let i = 0; i < action_names.length; i++) {
            const pct = (aggregate[i] * 100).toFixed(1);
            const item = document.createElement('div');
            item.className = 'strategy-legend-item';
            item.innerHTML = `
                <span class="strategy-legend-swatch" style="background:${colors[i]}"></span>
                <span class="strategy-legend-text">${action_names[i]} ${pct}%</span>
            `;
            el.strategyLegend.appendChild(item);
        }

        // Render table header
        el.strategyThead.innerHTML = '';
        const headerRow = document.createElement('tr');
        headerRow.innerHTML = `<th>Hand</th>` + action_names.map(n => `<th>${n}</th>`).join('');
        el.strategyThead.appendChild(headerRow);

        // Render table body (sorted by weight descending, limited to top 50)
        el.strategyTbody.innerHTML = '';
        const sortedHands = [...hands].sort((a, b) => b.weight - a.weight);
        const displayHands = sortedHands.slice(0, 50);

        for (const hand of displayHands) {
            const row = document.createElement('tr');
            let cells = `<td>${hand.combo}</td>`;
            for (let i = 0; i < hand.actions.length; i++) {
                const pct = (hand.actions[i] * 100).toFixed(0);
                const barWidth = Math.min(40, hand.actions[i] * 40);
                cells += `<td>
                    <span class="strategy-cell-bar" style="width:${barWidth}px;background:${colors[i]}"></span>
                    ${pct}%
                </td>`;
            }
            row.innerHTML = cells;
            el.strategyTbody.appendChild(row);
        }

    } catch (error) {
        console.error('Strategy display error:', error);
        el.strategySection.style.display = 'none';
    }
}

// === Start ===
initialize();
