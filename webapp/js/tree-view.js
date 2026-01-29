// Tree View module - Outline-style tree with Monker-like action labels

// State
let treeData = null;
let expandedNodes = new Set();
let selectedPath = '';
let onNodeSelect = null;
let treeViewEl = null;
let numPlayers = 2; // Default to 2 players

// Flat list of visible nodes for keyboard navigation
let visibleNodes = [];

/**
 * Initialize the tree view
 */
export function initTreeView(container, onSelect) {
    treeViewEl = container;
    onNodeSelect = onSelect;

    // Set up keyboard navigation
    container.addEventListener('keydown', handleKeyDown);
}

/**
 * Set the number of players (for display purposes)
 */
export function setNumPlayers(n) {
    numPlayers = n;
}

/**
 * Set tree data and render
 */
export function setTreeData(root) {
    treeData = root;
    expandedNodes.clear();
    expandedNodes.add(''); // Root always expanded
    selectedPath = '';
    renderTree();
}

/**
 * Clear the tree
 */
export function clearTree() {
    treeData = null;
    expandedNodes.clear();
    selectedPath = '';
    visibleNodes = [];
    if (treeViewEl) {
        treeViewEl.innerHTML = '<div class="tree-loading">Build tree to view</div>';
    }
}

/**
 * Get currently selected path
 */
export function getSelectedPath() {
    return selectedPath;
}

/**
 * Set selection and scroll into view
 */
export function setSelectedPath(path) {
    selectedPath = path;
    updateSelection();
    scrollToSelected();
}

/**
 * Toggle node expansion
 */
export async function toggleNode(path, getChildren) {
    if (expandedNodes.has(path)) {
        expandedNodes.delete(path);
        renderTree();
    } else {
        await expandNode(path, getChildren);
    }
}

/**
 * Expand a node
 */
export async function expandNode(path, getChildren) {
    if (!treeData) return;

    const node = findNode(treeData, path);
    if (!node || !node.has_children) return;

    // Load children if not loaded
    if (node.children.length === 0 && node.has_children) {
        const children = await getChildren(path);
        node.children = children || [];
    }

    expandedNodes.add(path);
    renderTree();
}

/**
 * Expand all nodes up to depth
 */
export async function expandAll(getChildren, maxDepth = 3) {
    if (!treeData) return;
    await expandRecursive(treeData, getChildren, 0, maxDepth);
    renderTree();
}

/**
 * Collapse all except root
 */
export function collapseAll() {
    expandedNodes.clear();
    expandedNodes.add('');
    renderTree();
}

/**
 * Expand path to show a specific node
 */
export async function expandToPath(targetPath, getChildren) {
    if (!treeData || !targetPath) return;

    const parts = targetPath.split('.');
    let currentPath = '';

    for (let i = 0; i < parts.length; i++) {
        await expandNode(currentPath, getChildren);
        currentPath = currentPath ? `${currentPath}.${parts[i]}` : parts[i];
    }

    renderTree();
}

// === Internal Functions ===

async function expandRecursive(node, getChildren, depth, maxDepth) {
    if (depth >= maxDepth || !node.has_children) return;

    if (node.children.length === 0 && node.has_children) {
        const children = await getChildren(node.path);
        node.children = children || [];
    }

    expandedNodes.add(node.path);

    for (const child of node.children) {
        await expandRecursive(child, getChildren, depth + 1, maxDepth);
    }
}

function findNode(root, path) {
    if (path === '' || path === root.path) return root;

    const parts = path.split('.');
    let current = root;

    for (const part of parts) {
        const idx = parseInt(part);
        if (!current.children || idx >= current.children.length) return null;
        current = current.children[idx];
    }

    return current;
}

function renderTree() {
    if (!treeViewEl || !treeData) return;

    visibleNodes = [];
    treeViewEl.innerHTML = '';

    const rootEl = renderNode(treeData, 0, null);
    treeViewEl.appendChild(rootEl);
}

function renderNode(node, depth, parentStreet) {
    const container = document.createElement('div');
    container.className = 'tree-node';
    container.dataset.path = node.path;

    // Track visible nodes for keyboard nav
    visibleNodes.push(node.path);

    // Main row
    const row = document.createElement('div');
    row.className = 'tree-row';
    if (node.path === selectedPath) {
        row.classList.add('selected');
    }

    // Caret for expand/collapse
    const caret = document.createElement('span');
    caret.className = 'tree-caret';
    if (node.has_children) {
        caret.classList.add('expandable');
        if (expandedNodes.has(node.path)) {
            caret.classList.add('expanded');
        }
        caret.innerHTML = '&#9654;'; // Right-pointing triangle
        caret.addEventListener('click', (e) => {
            e.stopPropagation();
            window.treeViewToggle(node.path);
        });
    }
    row.appendChild(caret);

    // Small icon for node type
    const icon = document.createElement('span');
    icon.className = `tree-icon ${node.node_type}`;
    row.appendChild(icon);

    // Action label (Monker-style)
    const label = document.createElement('span');
    label.className = 'tree-label';
    if (node.action_type) {
        label.classList.add(`action-${node.action_type}`);
    }
    label.textContent = formatActionLabel(node);
    row.appendChild(label);

    // Street badge (only on transitions)
    const nodeStreet = node.street?.toLowerCase();
    if (nodeStreet && nodeStreet !== parentStreet && depth > 0) {
        const streetBadge = document.createElement('span');
        streetBadge.className = `tree-street ${nodeStreet}`;
        streetBadge.textContent = node.street;
        row.appendChild(streetBadge);
    }

    // Node count (muted)
    if (node.subtree_size > 1) {
        const count = document.createElement('span');
        count.className = 'tree-count';
        count.textContent = formatCount(node.subtree_size);
        count.title = `${node.subtree_size} nodes in subtree`;
        row.appendChild(count);
    }

    // Click to select
    row.addEventListener('click', () => {
        if (onNodeSelect) {
            onNodeSelect(node.path);
        }
    });

    container.appendChild(row);

    // Children
    if (node.children && node.children.length > 0) {
        const childrenContainer = document.createElement('div');
        childrenContainer.className = 'tree-children';
        if (!expandedNodes.has(node.path)) {
            childrenContainer.classList.add('collapsed');
        }

        for (const child of node.children) {
            childrenContainer.appendChild(renderNode(child, depth + 1, nodeStreet));
        }

        container.appendChild(childrenContainer);
    }

    return container;
}

/**
 * Format action label in Monker style
 */
function formatActionLabel(node) {
    // Root node
    if (node.path === '') {
        const players = getPlayerCount(node);
        return `${players}-WAY, ${node.street || 'PREFLOP'}`;
    }

    // Terminal nodes - use label which includes "Action -> Result"
    if (node.node_type === 'terminal') {
        const label = node.label || '';
        // Format: "Fold -> BB wins" -> "FOLD → BB wins"
        if (label.toLowerCase().includes('fold')) {
            return `FOLD → ${node.terminal_result || 'wins'}`;
        }
        if (label.toLowerCase().includes('call')) {
            const match = label.match(/call\s*(\d+)/i);
            const amt = match ? match[1] : '';
            return `CALL ${amt} → ${node.terminal_result || 'Showdown'}`;
        }
        if (label.toLowerCase().includes('check')) {
            return `CHECK → ${node.terminal_result || 'Showdown'}`;
        }
        if (label.toLowerCase().includes('all-in')) {
            const match = label.match(/all-in\s*(\d+)/i);
            const amt = match ? match[1] : '';
            return `ALL-IN ${amt} → ${node.terminal_result || 'Showdown'}`;
        }
        // Fallback to terminal result
        return node.terminal_result || 'Terminal';
    }

    // Action labels
    const label = node.label || '';

    // Normalize action labels to Monker style
    if (label.toLowerCase().includes('fold')) return 'FOLD';
    if (label.toLowerCase().includes('check')) return 'CHECK';
    if (label.toLowerCase().includes('call')) {
        const match = label.match(/\d+/);
        return match ? `CALL ${match[0]}` : 'CALL';
    }
    if (label.toLowerCase().includes('all-in') || label.toLowerCase().includes('allin')) {
        const match = label.match(/\d+/);
        return match ? `ALL-IN ${match[0]}` : 'ALL-IN';
    }
    if (label.toLowerCase().includes('raise')) {
        const match = label.match(/\d+/);
        return match ? `RAISE ${match[0]}` : 'RAISE';
    }
    if (label.toLowerCase().includes('bet')) {
        const match = label.match(/\d+/);
        return match ? `BET ${match[0]}` : 'BET';
    }

    // Default: uppercase the label
    return label.toUpperCase();
}

function getPlayerCount(node) {
    // Use the module-level numPlayers set by main.js
    return numPlayers.toString();
}

function formatCount(n) {
    if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
    if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
    return n.toString();
}

function updateSelection() {
    if (!treeViewEl) return;

    // Remove old selection
    const oldSelected = treeViewEl.querySelector('.tree-row.selected');
    if (oldSelected) {
        oldSelected.classList.remove('selected');
    }

    // Add new selection
    const nodeEl = treeViewEl.querySelector(`[data-path="${selectedPath}"]`);
    if (nodeEl) {
        const row = nodeEl.querySelector('.tree-row');
        if (row) {
            row.classList.add('selected');
        }
    }
}

function scrollToSelected() {
    if (!treeViewEl) return;

    const nodeEl = treeViewEl.querySelector(`[data-path="${selectedPath}"]`);
    if (nodeEl) {
        nodeEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

// === Keyboard Navigation ===

function handleKeyDown(e) {
    if (!treeData || visibleNodes.length === 0) return;

    const currentIdx = visibleNodes.indexOf(selectedPath);

    switch (e.key) {
        case 'ArrowUp':
            e.preventDefault();
            if (currentIdx > 0) {
                selectNodeByIndex(currentIdx - 1);
            }
            break;

        case 'ArrowDown':
            e.preventDefault();
            if (currentIdx < visibleNodes.length - 1) {
                selectNodeByIndex(currentIdx + 1);
            }
            break;

        case 'ArrowLeft':
            e.preventDefault();
            if (expandedNodes.has(selectedPath)) {
                // Collapse current node
                expandedNodes.delete(selectedPath);
                renderTree();
            } else if (selectedPath) {
                // Go to parent
                const parts = selectedPath.split('.');
                parts.pop();
                const parentPath = parts.join('.');
                if (onNodeSelect) {
                    onNodeSelect(parentPath);
                }
            }
            break;

        case 'ArrowRight':
            e.preventDefault();
            const node = findNode(treeData, selectedPath);
            if (node && node.has_children) {
                if (!expandedNodes.has(selectedPath)) {
                    // Expand
                    window.treeViewToggle(selectedPath);
                } else if (node.children && node.children.length > 0) {
                    // Go to first child
                    if (onNodeSelect) {
                        onNodeSelect(node.children[0].path);
                    }
                }
            }
            break;

        case 'Enter':
        case ' ':
            e.preventDefault();
            const currentNode = findNode(treeData, selectedPath);
            if (currentNode && currentNode.has_children) {
                window.treeViewToggle(selectedPath);
            }
            break;
    }
}

function selectNodeByIndex(idx) {
    if (idx >= 0 && idx < visibleNodes.length) {
        const path = visibleNodes[idx];
        if (onNodeSelect) {
            onNodeSelect(path);
        }
    }
}
