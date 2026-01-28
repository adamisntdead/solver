// Tree View module for collapsible game tree visualization

// State
let treeData = null;
let expandedNodes = new Set();
let selectedPath = '';
let onNodeSelect = null;

// DOM elements
let treeViewEl = null;

/**
 * Initialize the tree view
 * @param {HTMLElement} container - The tree view container element
 * @param {Function} onSelect - Callback when a node is selected
 */
export function initTreeView(container, onSelect) {
    treeViewEl = container;
    onNodeSelect = onSelect;
}

/**
 * Set the tree data from WASM
 * @param {Object} root - The tree root node from get_tree_root()
 */
export function setTreeData(root) {
    treeData = root;
    expandedNodes.clear();
    // Auto-expand root
    expandedNodes.add('');
    renderTree();
}

/**
 * Clear the tree view
 */
export function clearTree() {
    treeData = null;
    expandedNodes.clear();
    selectedPath = '';
    if (treeViewEl) {
        treeViewEl.innerHTML = '<div class="tree-loading">Build tree to view</div>';
    }
}

/**
 * Set the selected path and highlight it
 * @param {string} path - The path to select
 */
export function setSelectedPath(path) {
    selectedPath = path;
    updateSelection();
    scrollToSelected();
}

/**
 * Expand all nodes (up to a reasonable depth)
 * @param {Function} getChildren - Function to get children from WASM
 * @param {number} maxDepth - Maximum depth to expand
 */
export async function expandAll(getChildren, maxDepth = 3) {
    if (!treeData) return;

    await expandNodeRecursive(treeData, getChildren, 0, maxDepth);
    renderTree();
}

/**
 * Collapse all nodes except root
 */
export function collapseAll() {
    expandedNodes.clear();
    expandedNodes.add(''); // Keep root expanded
    renderTree();
}

/**
 * Toggle a node's expanded state
 * @param {string} path - The node path
 * @param {Function} getChildren - Function to get children from WASM
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
 * Expand a specific node
 * @param {string} path - The node path
 * @param {Function} getChildren - Function to get children from WASM
 */
export async function expandNode(path, getChildren) {
    if (!treeData) return;

    const node = findNode(treeData, path);
    if (!node || !node.has_children) return;

    // Load children if not already loaded
    if (node.children.length === 0 && node.has_children) {
        const children = await getChildren(path);
        node.children = children;
    }

    expandedNodes.add(path);
    renderTree();
}

// === Internal Functions ===

async function expandNodeRecursive(node, getChildren, depth, maxDepth) {
    if (depth >= maxDepth || !node.has_children) return;

    const path = node.path;

    // Load children if needed
    if (node.children.length === 0 && node.has_children) {
        const children = await getChildren(path);
        node.children = children;
    }

    expandedNodes.add(path);

    // Recursively expand children
    for (const child of node.children) {
        await expandNodeRecursive(child, getChildren, depth + 1, maxDepth);
    }
}

function findNode(root, path) {
    if (path === '' || path === root.path) return root;

    const parts = path.split('.');
    let current = root;

    for (let i = 0; i < parts.length; i++) {
        const idx = parseInt(parts[i]);
        if (!current.children || idx >= current.children.length) {
            return null;
        }
        current = current.children[idx];
    }

    return current;
}

function renderTree() {
    if (!treeViewEl || !treeData) return;

    treeViewEl.innerHTML = '';
    const rootEl = renderNode(treeData, 0);
    treeViewEl.appendChild(rootEl);
}

function renderNode(node, depth) {
    const container = document.createElement('div');
    container.className = 'tree-node';
    container.dataset.path = node.path;

    // Row
    const row = document.createElement('div');
    row.className = 'tree-node-row';
    if (node.path === selectedPath) {
        row.classList.add('selected');
    }

    // Toggle button
    const toggle = document.createElement('span');
    toggle.className = 'tree-toggle';
    if (node.has_children) {
        toggle.classList.add('expandable');
        toggle.textContent = expandedNodes.has(node.path) ? '▼' : '▶';
        toggle.addEventListener('click', (e) => {
            e.stopPropagation();
            window.treeViewToggle(node.path);
        });
    }
    row.appendChild(toggle);

    // Icon
    const icon = document.createElement('span');
    icon.className = `tree-node-icon ${node.node_type}`;
    if (node.node_type === 'player') {
        icon.textContent = node.player_name ? node.player_name[0] : 'P';
    } else if (node.node_type === 'terminal') {
        icon.textContent = '●';
    }
    row.appendChild(icon);

    // Label
    const label = document.createElement('span');
    label.className = 'tree-node-label';
    if (node.action_type) {
        label.classList.add(`action-${node.action_type}`);
    }
    label.textContent = node.label;
    row.appendChild(label);

    // Street badge (only show on street changes or terminals)
    if (node.street && (depth === 0 || node.node_type === 'terminal')) {
        const street = document.createElement('span');
        street.className = `tree-node-street ${node.street.toLowerCase()}`;
        street.textContent = node.street;
        row.appendChild(street);
    }

    // Node count
    if (node.subtree_size > 1) {
        const count = document.createElement('span');
        count.className = 'tree-node-count';
        count.textContent = `(${formatCount(node.subtree_size)})`;
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
            childrenContainer.appendChild(renderNode(child, depth + 1));
        }

        container.appendChild(childrenContainer);
    }

    return container;
}

function updateSelection() {
    if (!treeViewEl) return;

    // Remove old selection
    const oldSelected = treeViewEl.querySelector('.tree-node-row.selected');
    if (oldSelected) {
        oldSelected.classList.remove('selected');
    }

    // Add new selection
    const node = treeViewEl.querySelector(`[data-path="${selectedPath}"]`);
    if (node) {
        const row = node.querySelector('.tree-node-row');
        if (row) {
            row.classList.add('selected');
        }
    }
}

function scrollToSelected() {
    if (!treeViewEl) return;

    const node = treeViewEl.querySelector(`[data-path="${selectedPath}"]`);
    if (node) {
        node.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

function formatCount(n) {
    if (n >= 1000000) {
        return (n / 1000000).toFixed(1) + 'M';
    } else if (n >= 1000) {
        return (n / 1000).toFixed(1) + 'K';
    }
    return n.toString();
}

/**
 * Expand the path to show a specific node
 * @param {string} targetPath - The path to expand to
 * @param {Function} getChildren - Function to get children from WASM
 */
export async function expandToPath(targetPath, getChildren) {
    if (!treeData || !targetPath) return;

    const parts = targetPath.split('.');
    let currentPath = '';

    for (let i = 0; i < parts.length; i++) {
        // Expand the current path
        await expandNode(currentPath, getChildren);

        // Build next path
        if (currentPath === '') {
            currentPath = parts[i];
        } else {
            currentPath = currentPath + '.' + parts[i];
        }
    }

    renderTree();
}
