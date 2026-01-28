// Main entry point for the Poker Tree Builder webapp

import init, {
    build_tree,
    validate_config,
    parse_bet_size_preview,
    get_node_at_path,
    get_action_result,
    get_default_config,
    get_tree_root,
    get_tree_children
} from '../pkg/solver_wasm.js';

import * as TreeView from './tree-view.js';

// Global state
let wasmLoaded = false;
let currentConfig = null;
let currentPath = '';
let treeBuilt = false;

// === DOM Elements ===
const elements = {
    // Config inputs
    spotName: document.getElementById('spot-name'),
    numPlayers: document.getElementById('num-players'),
    stackSize: document.getElementById('stack-size'),
    betType: document.getElementById('bet-type'),
    sbSize: document.getElementById('sb-size'),
    bbSize: document.getElementById('bb-size'),
    preflopOpenBet: document.getElementById('preflop-open-bet'),
    preflopOpenRaise: document.getElementById('preflop-open-raise'),
    preflop3betRaise: document.getElementById('preflop-3bet-raise'),
    flopBet: document.getElementById('flop-bet'),
    flopRaise: document.getElementById('flop-raise'),
    turnBet: document.getElementById('turn-bet'),
    turnRaise: document.getElementById('turn-raise'),
    riverBet: document.getElementById('river-bet'),
    riverRaise: document.getElementById('river-raise'),
    maxRaises: document.getElementById('max-raises'),

    // Buttons
    buildBtn: document.getElementById('build-btn'),
    saveBtn: document.getElementById('save-btn'),
    loadBtn: document.getElementById('load-btn'),
    loadInput: document.getElementById('load-input'),
    backBtn: document.getElementById('back-btn'),
    resetBtn: document.getElementById('reset-btn'),

    // Tree view
    treeView: document.getElementById('tree-view'),
    expandAllBtn: document.getElementById('expand-all-btn'),
    collapseAllBtn: document.getElementById('collapse-all-btn'),

    // Tree navigator
    streetBadge: document.getElementById('street-badge'),
    potValue: document.getElementById('pot-value'),
    playerInfo: document.getElementById('player-info'),
    playerBadge: document.getElementById('player-badge'),
    stackValue: document.getElementById('stack-value'),
    actionButtons: document.getElementById('action-buttons'),
    actionsContainer: document.getElementById('actions-container'),
    terminalResult: document.getElementById('terminal-result'),
    terminalText: document.getElementById('terminal-text'),
    historyList: document.getElementById('history-list'),
    breadcrumb: document.getElementById('breadcrumb'),

    // Stats
    statNodes: document.getElementById('stat-nodes'),
    statTerminals: document.getElementById('stat-terminals'),
    statPlayerNodes: document.getElementById('stat-player-nodes'),
    statDepth: document.getElementById('stat-depth'),
    statInfosets: document.getElementById('stat-infosets'),
    statMemoryCompressed: document.getElementById('stat-memory-compressed'),
    statMemoryUncompressed: document.getElementById('stat-memory-uncompressed'),
    statPathDepth: document.getElementById('stat-path-depth'),
    statEffectiveStack: document.getElementById('stat-effective-stack'),

    // Preview
    previewSize: document.getElementById('preview-size'),
    previewPot: document.getElementById('preview-pot'),
    previewStack: document.getElementById('preview-stack'),
    previewResult: document.getElementById('preview-result'),

    // Status
    statusBar: document.getElementById('status-bar')
};

// === Initialize ===
async function initialize() {
    setStatus('Loading WASM module...');

    try {
        await init();
        wasmLoaded = true;
        setStatus('Ready', 'success');

        // Initialize tree view
        TreeView.initTreeView(elements.treeView, handleTreeNodeSelect);

        // Expose toggle function for tree view clicks
        window.treeViewToggle = (path) => {
            TreeView.toggleNode(path, getTreeChildren);
        };

        // Load default config
        const defaultConfig = get_default_config();
        currentConfig = JSON.parse(defaultConfig);
        loadConfigToUI(currentConfig);

        // Set up event listeners
        setupEventListeners();

        // Auto-build tree
        buildTree();
    } catch (error) {
        setStatus(`Failed to load WASM: ${error.message}`, 'error');
        console.error('WASM init error:', error);
    }
}

// === Event Listeners ===
function setupEventListeners() {
    // Build button
    elements.buildBtn.addEventListener('click', buildTree);

    // Save/Load buttons
    elements.saveBtn.addEventListener('click', saveConfig);
    elements.loadBtn.addEventListener('click', () => elements.loadInput.click());
    elements.loadInput.addEventListener('change', loadConfig);

    // Navigation buttons
    elements.backBtn.addEventListener('click', navigateBack);
    elements.resetBtn.addEventListener('click', resetNavigation);

    // Tree view controls
    elements.expandAllBtn.addEventListener('click', handleExpandAll);
    elements.collapseAllBtn.addEventListener('click', handleCollapseAll);

    // Bet size preview
    elements.previewSize.addEventListener('input', updateBetSizePreview);
    elements.previewPot.addEventListener('input', updateBetSizePreview);
    elements.previewStack.addEventListener('input', updateBetSizePreview);
}

// === Build Tree ===
async function buildTree() {
    if (!wasmLoaded) {
        setStatus('WASM not loaded', 'error');
        return;
    }

    setStatus('Building tree...');

    try {
        // Collect config from UI
        currentConfig = collectConfigFromUI();
        const configJson = JSON.stringify(currentConfig);

        // Validate first
        const validation = validate_config(configJson);
        if (!validation.valid) {
            setStatus(`Validation errors: ${validation.errors.join(', ')}`, 'error');
            return;
        }

        // Build tree
        const result = build_tree(configJson);

        if (!result.success) {
            setStatus(`Build failed: ${result.error}`, 'error');
            return;
        }

        // Update stats display
        updateStats(result.stats, result.memory);

        treeBuilt = true;
        currentPath = '';

        // Load tree view
        await loadTreeView();

        // Navigate to root
        navigateToPath('');

        setStatus('Tree built successfully', 'success');
    } catch (error) {
        setStatus(`Build error: ${error.message}`, 'error');
        console.error('Build error:', error);
    }
}

// === Config Management ===
function collectConfigFromUI() {
    const sb = parseInt(elements.sbSize.value) || 1;
    const bb = parseInt(elements.bbSize.value) || 2;
    const stack = parseInt(elements.stackSize.value) || 100;

    return {
        name: elements.spotName.value,
        num_players: parseInt(elements.numPlayers.value),
        starting_stacks: [stack * bb],
        bet_type: elements.betType.value,
        preflop: {
            blinds: [sb, bb],
            ante: 0,
            bb_ante: 0,
            open_sizes: {
                bet: elements.preflopOpenBet.value,
                raise: elements.preflopOpenRaise.value
            },
            three_bet_sizes: {
                bet: '',
                raise: elements.preflop3betRaise.value
            },
            four_bet_sizes: {
                bet: '',
                raise: '2.2x, a'
            },
            allow_limps: true
        },
        flop: {
            sizes: {
                bet: elements.flopBet.value,
                raise: elements.flopRaise.value
            }
        },
        turn: {
            sizes: {
                bet: elements.turnBet.value,
                raise: elements.turnRaise.value
            }
        },
        river: {
            sizes: {
                bet: elements.riverBet.value,
                raise: elements.riverRaise.value
            }
        },
        max_raises_per_round: parseInt(elements.maxRaises.value) || 4,
        force_all_in_threshold: 0.15,
        merge_threshold: 0.1,
        add_all_in_threshold: 1.5
    };
}

function loadConfigToUI(config) {
    elements.spotName.value = config.name || 'HU 100bb';
    elements.numPlayers.value = config.num_players || 2;

    const bb = config.preflop?.blinds?.[1] || 2;
    const stack = config.starting_stacks?.[0] || 200;
    elements.stackSize.value = Math.round(stack / bb);

    elements.betType.value = config.bet_type || 'NoLimit';
    elements.sbSize.value = config.preflop?.blinds?.[0] || 1;
    elements.bbSize.value = bb;

    elements.preflopOpenBet.value = config.preflop?.open_sizes?.bet || '';
    elements.preflopOpenRaise.value = config.preflop?.open_sizes?.raise || '';
    elements.preflop3betRaise.value = config.preflop?.three_bet_sizes?.raise || '';

    elements.flopBet.value = config.flop?.sizes?.bet || '';
    elements.flopRaise.value = config.flop?.sizes?.raise || '';
    elements.turnBet.value = config.turn?.sizes?.bet || '';
    elements.turnRaise.value = config.turn?.sizes?.raise || '';
    elements.riverBet.value = config.river?.sizes?.bet || '';
    elements.riverRaise.value = config.river?.sizes?.raise || '';

    elements.maxRaises.value = config.max_raises_per_round || 4;
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
            buildTree();
            setStatus('Config loaded', 'success');
        } catch (error) {
            setStatus(`Failed to load config: ${error.message}`, 'error');
        }
    };
    reader.readAsText(file);

    // Reset input so same file can be loaded again
    event.target.value = '';
}

// === Tree Navigation ===
function navigateToPath(path) {
    if (!wasmLoaded || !treeBuilt) return;

    try {
        const configJson = JSON.stringify(currentConfig);
        const nodeState = get_node_at_path(configJson, path);

        if (nodeState.error) {
            setStatus(`Navigation error: ${nodeState.error}`, 'error');
            return;
        }

        currentPath = path;
        updateNodeDisplay(nodeState);
        updateNavigationControls();

        // Sync tree view selection
        TreeView.setSelectedPath(path);

    } catch (error) {
        setStatus(`Navigation error: ${error.message}`, 'error');
        console.error('Navigation error:', error);
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
        updateNodeDisplay(nodeState);
        updateNavigationControls();

        // Sync tree view - expand path and select
        await TreeView.expandToPath(currentPath, getTreeChildren);
        TreeView.setSelectedPath(currentPath);

    } catch (error) {
        setStatus(`Action error: ${error.message}`, 'error');
        console.error('Action error:', error);
    }
}

function navigateBack() {
    if (!currentPath) return;

    const parts = currentPath.split('.');
    parts.pop();
    navigateToPath(parts.join('.'));
}

function resetNavigation() {
    navigateToPath('');
}

function jumpToHistoryPoint(path) {
    navigateToPath(path);
}

// === UI Updates ===
function updateNodeDisplay(nodeState) {
    // Update street badge
    const streetClass = nodeState.street.toLowerCase();
    elements.streetBadge.textContent = nodeState.street;
    elements.streetBadge.className = `street-badge ${streetClass}`;

    // Update pot
    elements.potValue.textContent = nodeState.pot;

    // Handle different node types
    if (nodeState.node_type === 'player') {
        elements.playerInfo.style.display = 'flex';
        elements.actionsContainer.style.display = 'block';
        elements.terminalResult.style.display = 'none';

        elements.playerBadge.textContent = nodeState.player_name;

        // Show current player's stack
        const playerStack = nodeState.stacks[nodeState.player_to_act];
        elements.stackValue.textContent = playerStack;

        // Render action buttons
        renderActionButtons(nodeState.actions);

    } else if (nodeState.node_type === 'terminal') {
        elements.playerInfo.style.display = 'none';
        elements.actionsContainer.style.display = 'none';
        elements.terminalResult.style.display = 'flex';

        elements.terminalText.textContent = nodeState.terminal_result;
    }

    // Update history
    renderHistory(nodeState.action_history);

    // Update breadcrumb
    renderBreadcrumb(nodeState.action_history);

    // Update path depth stat
    const pathDepth = currentPath ? currentPath.split('.').length : 0;
    elements.statPathDepth.textContent = pathDepth;
}

function renderActionButtons(actions) {
    elements.actionButtons.innerHTML = '';

    for (const action of actions) {
        const btn = document.createElement('button');
        btn.className = `action-btn action-${action.action_type}`;
        btn.textContent = action.name;
        btn.addEventListener('click', () => handleActionClick(action.index));
        elements.actionButtons.appendChild(btn);
    }
}

function renderHistory(history) {
    elements.historyList.innerHTML = '';

    if (history.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'history-empty';
        empty.textContent = 'No actions yet';
        empty.style.color = 'var(--text-secondary)';
        empty.style.fontStyle = 'italic';
        elements.historyList.appendChild(empty);
        return;
    }

    for (const item of history) {
        const div = document.createElement('div');
        div.className = 'history-item';
        div.addEventListener('click', () => jumpToHistoryPoint(item.path));

        const player = document.createElement('span');
        player.className = 'history-player';
        player.textContent = item.player_name;

        const action = document.createElement('span');
        action.className = 'history-action';
        action.textContent = item.action;

        div.appendChild(player);
        div.appendChild(action);
        elements.historyList.appendChild(div);
    }
}

function renderBreadcrumb(history) {
    elements.breadcrumb.innerHTML = '';

    // Root crumb
    const rootCrumb = document.createElement('span');
    rootCrumb.className = `crumb ${history.length === 0 ? 'active' : ''}`;
    rootCrumb.textContent = 'Root';
    rootCrumb.addEventListener('click', resetNavigation);
    elements.breadcrumb.appendChild(rootCrumb);

    // Add crumbs for each action
    for (let i = 0; i < history.length; i++) {
        const item = history[i];
        const isLast = i === history.length - 1;

        // Separator
        const sep = document.createElement('span');
        sep.className = 'crumb-separator';
        sep.textContent = '>';
        elements.breadcrumb.appendChild(sep);

        // Crumb
        const crumb = document.createElement('span');
        crumb.className = `crumb ${isLast ? 'active' : ''}`;
        crumb.textContent = `${item.player_name} ${item.action}`;

        // Build path to this point
        const pathParts = currentPath.split('.');
        const pathToHere = pathParts.slice(0, i + 1).join('.');
        crumb.addEventListener('click', () => navigateToPath(pathToHere));

        elements.breadcrumb.appendChild(crumb);
    }
}

function updateNavigationControls() {
    elements.backBtn.disabled = !currentPath;
}

function updateStats(stats, memory) {
    if (stats) {
        elements.statNodes.textContent = stats.node_count.toLocaleString();
        elements.statTerminals.textContent = stats.terminal_count.toLocaleString();
        elements.statPlayerNodes.textContent = stats.player_node_count.toLocaleString();
        elements.statDepth.textContent = stats.max_depth;
    }

    if (memory) {
        elements.statInfosets.textContent = memory.info_set_count.toLocaleString();
        elements.statMemoryCompressed.textContent = formatBytes(memory.compressed_bytes);
        elements.statMemoryUncompressed.textContent = formatBytes(memory.uncompressed_bytes);
    }

    // Update effective stack
    const bb = parseInt(elements.bbSize.value) || 2;
    const stack = parseInt(elements.stackSize.value) || 100;
    elements.statEffectiveStack.textContent = `${stack} BB`;
}

// === Bet Size Preview ===
function updateBetSizePreview() {
    if (!wasmLoaded) return;

    const sizeStr = elements.previewSize.value.trim();
    if (!sizeStr) {
        elements.previewResult.textContent = 'Enter a bet size to preview';
        elements.previewResult.className = 'preview-result';
        return;
    }

    const pot = parseInt(elements.previewPot.value) || 100;
    const stack = parseInt(elements.previewStack.value) || 500;

    try {
        const result = parse_bet_size_preview(sizeStr, pot, stack);

        if (result.valid) {
            const pctStr = result.pot_percentage !== null
                ? ` (${result.pot_percentage.toFixed(1)}% pot)`
                : '';
            elements.previewResult.textContent = `${result.chips} chips${pctStr}`;
            elements.previewResult.className = 'preview-result valid';
        } else {
            elements.previewResult.textContent = result.error;
            elements.previewResult.className = 'preview-result invalid';
        }
    } catch (error) {
        elements.previewResult.textContent = error.message;
        elements.previewResult.className = 'preview-result invalid';
    }
}

// === Tree View Helpers ===
async function getTreeChildren(path) {
    if (!wasmLoaded || !currentConfig) return [];

    try {
        const configJson = JSON.stringify(currentConfig);
        const children = get_tree_children(configJson, path);
        return children || [];
    } catch (error) {
        console.error('Error getting tree children:', error);
        return [];
    }
}

function handleTreeNodeSelect(path) {
    // Navigate to the selected node (this also syncs tree view selection)
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

// === Utilities ===
function formatBytes(bytes) {
    if (typeof bytes !== 'number') return '-';

    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;

    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex++;
    }

    return `${size.toFixed(2)} ${units[unitIndex]}`;
}

function setStatus(message, type = '') {
    elements.statusBar.textContent = message;
    elements.statusBar.className = `status-bar ${type}`;
}

// === Start ===
initialize();
