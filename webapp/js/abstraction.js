// Hand Abstraction Explorer

import init, {
    compute_board_ehs,
    compute_board_winsplit,
    get_aggsi_buckets,
    compute_emd_histogram,
    lookup_hand
} from '../pkg/solver_wasm.js';

// === State ===
let wasmLoaded = false;
let currentBoard = '';
let currentHands = [];
let currentBuckets = [];
let selectedHand = null;
let activeView = 'hands';

// === DOM Elements ===
const el = {
    boardInput: document.getElementById('board-input'),
    boardCards: document.getElementById('board-cards'),
    cardPicker: document.getElementById('card-picker'),
    handLookupInput: document.getElementById('hand-lookup-input'),
    handLookupResult: document.getElementById('hand-lookup-result'),
    lookupCombo: document.getElementById('lookup-combo'),
    lookupEhs: document.getElementById('lookup-ehs'),
    lookupBucket: document.getElementById('lookup-bucket'),
    statValidHands: document.getElementById('stat-valid-hands'),
    statAvgEhs: document.getElementById('stat-avg-ehs'),
    statAggsiBuckets: document.getElementById('stat-aggsi-buckets'),
    statStreet: document.getElementById('stat-street'),
    sortBy: document.getElementById('sort-by'),
    viewHands: document.getElementById('view-hands'),
    viewBuckets: document.getElementById('view-buckets'),
    viewHistogram: document.getElementById('view-histogram'),
    statusBar: document.getElementById('status-bar')
};

// === Initialize ===
async function initialize() {
    setStatus('Loading WASM module...');

    try {
        await init();
        wasmLoaded = true;
        setStatus('Ready');

        setupEventListeners();
        setupCardPicker();
    } catch (err) {
        setStatus(`Failed to load WASM: ${err}`, 'error');
        console.error('WASM init failed:', err);
    }
}

function setStatus(message, type = '') {
    el.statusBar.textContent = message;
    el.statusBar.className = 'status-bar';
    if (type) {
        el.statusBar.classList.add(`status-${type}`);
    }
}

// === Event Listeners ===
function setupEventListeners() {
    // Board input
    el.boardInput.addEventListener('input', debounce(onBoardInput, 300));
    el.boardInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            onBoardInput();
        }
    });

    // Hand lookup
    el.handLookupInput.addEventListener('input', debounce(onHandLookup, 300));
    el.handLookupInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            onHandLookup();
        }
    });

    // Sort by
    el.sortBy.addEventListener('change', () => {
        sortAndRenderHands();
    });

    // View tabs
    document.querySelectorAll('.view-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            switchView(tab.dataset.view);
        });
    });

    // Board card clicks
    el.boardCards.querySelectorAll('.board-card').forEach(card => {
        card.addEventListener('click', (e) => {
            openCardPicker(e, card);
        });
    });

    // Close card picker on outside click
    document.addEventListener('click', (e) => {
        if (!el.cardPicker.contains(e.target) && !e.target.classList.contains('board-card')) {
            closeCardPicker();
        }
    });
}

function debounce(fn, ms) {
    let timer;
    return (...args) => {
        clearTimeout(timer);
        timer = setTimeout(() => fn(...args), ms);
    };
}

// === Card Picker ===
function setupCardPicker() {
    const ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'];
    const suits = [
        { char: 'h', name: 'hearts', symbol: '♥' },
        { char: 'd', name: 'diamonds', symbol: '♦' },
        { char: 'c', name: 'clubs', symbol: '♣' },
        { char: 's', name: 'spades', symbol: '♠' }
    ];

    let html = '';
    for (const suit of suits) {
        html += `<div class="suit-row-label">${suit.symbol}</div>`;
        html += '<div class="card-picker-grid">';
        for (const rank of ranks) {
            const card = rank + suit.char;
            html += `<button class="card-picker-btn ${suit.name}" data-card="${card}">${rank}</button>`;
        }
        html += '</div>';
    }

    el.cardPicker.innerHTML = html;

    el.cardPicker.querySelectorAll('.card-picker-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            selectCard(btn.dataset.card);
        });
    });
}

let activeCardSlot = null;

function openCardPicker(event, cardElement) {
    event.stopPropagation();
    activeCardSlot = cardElement;

    const rect = cardElement.getBoundingClientRect();
    el.cardPicker.style.top = `${rect.bottom + 4}px`;
    el.cardPicker.style.left = `${rect.left}px`;
    el.cardPicker.classList.add('active');

    // Mark used cards
    updateCardPickerState();
}

function closeCardPicker() {
    el.cardPicker.classList.remove('active');
    activeCardSlot = null;
}

function updateCardPickerState() {
    const usedCards = getUsedCards();
    el.cardPicker.querySelectorAll('.card-picker-btn').forEach(btn => {
        btn.classList.toggle('selected', usedCards.has(btn.dataset.card));
    });
}

function getUsedCards() {
    const cards = new Set();
    el.boardCards.querySelectorAll('.board-card').forEach(card => {
        if (card.dataset.card) {
            cards.add(card.dataset.card);
        }
    });
    return cards;
}

function selectCard(card) {
    if (!activeCardSlot) return;

    // Check if card is already used
    const usedCards = getUsedCards();
    if (usedCards.has(card) && activeCardSlot.dataset.card !== card) {
        return;
    }

    activeCardSlot.dataset.card = card;
    activeCardSlot.textContent = formatCardDisplay(card);
    activeCardSlot.classList.add('filled');
    activeCardSlot.classList.remove('hearts', 'diamonds', 'clubs', 'spades');
    activeCardSlot.classList.add(getSuitClass(card[1]));

    closeCardPicker();
    updateBoardFromCards();
}

function formatCardDisplay(card) {
    const suitSymbols = { h: '♥', d: '♦', c: '♣', s: '♠' };
    return card[0] + suitSymbols[card[1]];
}

function getSuitClass(suit) {
    const classes = { h: 'hearts', d: 'diamonds', c: 'clubs', s: 'spades' };
    return classes[suit] || '';
}

function updateBoardFromCards() {
    let board = '';
    el.boardCards.querySelectorAll('.board-card').forEach(card => {
        if (card.dataset.card) {
            board += card.dataset.card;
        }
    });
    el.boardInput.value = board;
    onBoardInput();
}

function updateCardsFromBoard(boardStr) {
    // Parse board string and update card display
    const cards = parseBoardString(boardStr);
    el.boardCards.querySelectorAll('.board-card').forEach((card, i) => {
        if (i < cards.length) {
            card.dataset.card = cards[i];
            card.textContent = formatCardDisplay(cards[i]);
            card.classList.add('filled');
            card.classList.remove('hearts', 'diamonds', 'clubs', 'spades');
            card.classList.add(getSuitClass(cards[i][1]));
        } else {
            delete card.dataset.card;
            card.textContent = '';
            card.classList.remove('filled', 'hearts', 'diamonds', 'clubs', 'spades');
        }
    });
}

function parseBoardString(str) {
    const cards = [];
    const s = str.trim();
    for (let i = 0; i < s.length; i += 2) {
        if (i + 1 < s.length) {
            cards.push(s[i].toUpperCase() + s[i + 1].toLowerCase());
        }
    }
    return cards;
}

// === Board Input ===
function onBoardInput() {
    const board = el.boardInput.value.trim();

    if (board.length >= 6) { // At least 3 cards (flop)
        updateCardsFromBoard(board);
        loadBoardData(board);
    } else {
        clearResults();
    }
}

async function loadBoardData(board) {
    if (!wasmLoaded) return;

    currentBoard = board;
    setStatus('Computing EHS...');

    try {
        // Compute EHS for all hands
        const ehsResult = compute_board_ehs(board);

        if (!ehsResult.success) {
            setStatus(`Error: ${ehsResult.error}`, 'error');
            return;
        }

        currentHands = ehsResult.hands;

        // Get AggSI buckets
        const bucketsResult = get_aggsi_buckets(board);
        if (bucketsResult.success) {
            currentBuckets = bucketsResult.buckets;
        }

        // Update stats
        updateStats(ehsResult.hands, bucketsResult);

        // Render current view
        sortAndRenderHands();
        renderBuckets();

        setStatus(`Loaded ${currentHands.length} hands`);
    } catch (err) {
        setStatus(`Error: ${err}`, 'error');
        console.error('Load board data error:', err);
    }
}

function updateStats(hands, bucketsResult) {
    el.statValidHands.textContent = hands.length;

    const avgEhs = hands.reduce((sum, h) => sum + h.ehs, 0) / hands.length;
    el.statAvgEhs.textContent = avgEhs.toFixed(3);

    if (bucketsResult.success) {
        el.statAggsiBuckets.textContent = bucketsResult.buckets.length;
    }

    // Determine street from board length
    const numCards = currentBoard.length / 2;
    const streets = { 3: 'Flop', 4: 'Turn', 5: 'River' };
    el.statStreet.textContent = streets[numCards] || '-';
}

function clearResults() {
    currentHands = [];
    currentBuckets = [];
    el.statValidHands.textContent = '-';
    el.statAvgEhs.textContent = '-';
    el.statAggsiBuckets.textContent = '-';
    el.statStreet.textContent = '-';

    el.viewHands.innerHTML = `
        <div class="empty-state">
            <div class="empty-state-icon">&#x1F0A1;</div>
            <div>Enter a board to explore hand abstractions</div>
        </div>
    `;

    el.viewBuckets.innerHTML = `
        <div class="empty-state">
            <div class="empty-state-icon">&#x1F4E6;</div>
            <div>Enter a board to view AggSI buckets</div>
        </div>
    `;
}

// === Hand Lookup ===
function onHandLookup() {
    const combo = el.handLookupInput.value.trim();

    if (combo.length !== 4 || !currentBoard) {
        el.handLookupResult.classList.remove('visible');
        return;
    }

    try {
        const result = lookup_hand(currentBoard, combo);

        if (!result.success) {
            el.lookupCombo.textContent = combo;
            el.lookupEhs.textContent = result.error || 'Invalid';
            el.lookupBucket.textContent = '-';
        } else {
            el.lookupCombo.textContent = result.combo;
            el.lookupEhs.textContent = result.ehs.toFixed(4);
            el.lookupBucket.textContent = result.aggsi_bucket !== null ? result.aggsi_bucket : '-';
        }

        el.handLookupResult.classList.add('visible');
    } catch (err) {
        console.error('Hand lookup error:', err);
    }
}

// === View Switching ===
function switchView(view) {
    activeView = view;

    document.querySelectorAll('.view-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.view === view);
    });

    document.querySelectorAll('.view-container').forEach(container => {
        container.classList.toggle('active', container.id === `view-${view}`);
    });
}

// === Render Functions ===
function sortAndRenderHands() {
    const sortBy = el.sortBy.value;

    let sorted = [...currentHands];
    switch (sortBy) {
        case 'ehs-desc':
            sorted.sort((a, b) => b.ehs - a.ehs);
            break;
        case 'ehs-asc':
            sorted.sort((a, b) => a.ehs - b.ehs);
            break;
        case 'combo':
            sorted.sort((a, b) => a.combo.localeCompare(b.combo));
            break;
        case 'bucket':
            // Sort by AggSI bucket, need to look up
            // For now just use EHS
            sorted.sort((a, b) => b.ehs - a.ehs);
            break;
    }

    renderHands(sorted);
}

function renderHands(hands) {
    if (hands.length === 0) {
        el.viewHands.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">&#x1F0A1;</div>
                <div>Enter a board to explore hand abstractions</div>
            </div>
        `;
        return;
    }

    let html = '<div class="hands-grid">';

    for (const hand of hands) {
        const ehsClass = getEhsClass(hand.ehs);
        const isSelected = selectedHand === hand.combo;

        html += `
            <div class="hand-card ${isSelected ? 'selected' : ''}" data-combo="${hand.combo}">
                <div class="hand-combo">${formatComboDisplay(hand.combo)}</div>
                <div class="hand-ehs ${ehsClass}">${(hand.ehs * 100).toFixed(1)}%</div>
            </div>
        `;
    }

    html += '</div>';
    el.viewHands.innerHTML = html;

    // Add click handlers
    el.viewHands.querySelectorAll('.hand-card').forEach(card => {
        card.addEventListener('click', () => {
            selectHandForHistogram(card.dataset.combo);
        });
    });
}

function formatComboDisplay(combo) {
    // Format like "A♠K♠"
    const suitSymbols = { h: '♥', d: '♦', c: '♣', s: '♠' };
    const suitColors = { h: 'hearts', d: 'diamonds', c: 'clubs', s: 'spades' };

    const r1 = combo[0];
    const s1 = combo[1];
    const r2 = combo[2];
    const s2 = combo[3];

    return `<span class="${suitColors[s1]}">${r1}${suitSymbols[s1]}</span><span class="${suitColors[s2]}">${r2}${suitSymbols[s2]}</span>`;
}

function getEhsClass(ehs) {
    if (ehs < 0.2) return 'ehs-very-low';
    if (ehs < 0.4) return 'ehs-low';
    if (ehs < 0.6) return 'ehs-medium';
    if (ehs < 0.8) return 'ehs-high';
    return 'ehs-very-high';
}

function renderBuckets() {
    if (currentBuckets.length === 0) {
        el.viewBuckets.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">&#x1F4E6;</div>
                <div>Enter a board to view AggSI buckets</div>
            </div>
        `;
        return;
    }

    let html = '<div class="bucket-list">';

    for (const bucket of currentBuckets) {
        html += `
            <div class="bucket-item" data-bucket="${bucket.bucket_id}">
                <span class="bucket-id">Bucket ${bucket.bucket_id}</span>
                <span class="bucket-count">${bucket.count} hands</span>
            </div>
        `;
    }

    html += '</div>';
    html += '<div class="bucket-hands" id="bucket-hands"></div>';

    el.viewBuckets.innerHTML = html;

    // Add click handlers
    el.viewBuckets.querySelectorAll('.bucket-item').forEach(item => {
        item.addEventListener('click', () => {
            showBucketHands(parseInt(item.dataset.bucket));

            // Update selection UI
            el.viewBuckets.querySelectorAll('.bucket-item').forEach(i => i.classList.remove('selected'));
            item.classList.add('selected');
        });
    });
}

function showBucketHands(bucketId) {
    const bucket = currentBuckets.find(b => b.bucket_id === bucketId);
    if (!bucket) return;

    const handsContainer = document.getElementById('bucket-hands');
    let html = `<h3>Hands in Bucket ${bucketId}</h3>`;
    html += '<div class="bucket-hands-grid">';

    for (const hand of bucket.hands) {
        html += `<span class="bucket-hand-chip">${formatComboDisplay(hand)}</span>`;
    }

    html += '</div>';
    handsContainer.innerHTML = html;
}

// === EMD Histogram ===
async function selectHandForHistogram(combo) {
    selectedHand = combo;

    // Update selection UI
    el.viewHands.querySelectorAll('.hand-card').forEach(card => {
        card.classList.toggle('selected', card.dataset.combo === combo);
    });

    // Check if board is non-river (EMD only works for flop/turn)
    const numCards = currentBoard.length / 2;
    if (numCards >= 5) {
        el.viewHistogram.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">&#x1F4CA;</div>
                <div>EMD histogram is only available for flop and turn boards</div>
            </div>
        `;
        return;
    }

    try {
        const result = compute_emd_histogram(currentBoard, combo);

        if (!result.success) {
            el.viewHistogram.innerHTML = `
                <div class="empty-state">
                    <div>Error: ${result.error}</div>
                </div>
            `;
            return;
        }

        renderHistogram(combo, result.histogram);
    } catch (err) {
        console.error('EMD histogram error:', err);
    }
}

function renderHistogram(combo, histogram) {
    const maxVal = Math.max(...histogram, 0.001);

    let html = `<h3 style="margin-bottom: 12px;">EMD Histogram for ${formatComboDisplay(combo)}</h3>`;
    html += '<div class="emd-histogram">';
    html += '<div class="histogram-container">';

    for (let i = 0; i < histogram.length; i++) {
        const height = (histogram[i] / maxVal) * 100;
        const equity = (i / (histogram.length - 1) * 100).toFixed(0);
        html += `<div class="histogram-bar" style="height: ${height}%" title="Equity ${equity}%: ${(histogram[i] * 100).toFixed(1)}%"></div>`;
    }

    html += '</div>';
    html += '<div class="histogram-labels">';
    html += '<span>0%</span>';
    html += '<span>25%</span>';
    html += '<span>50%</span>';
    html += '<span>75%</span>';
    html += '<span>100%</span>';
    html += '</div>';
    html += '</div>';

    // Add summary stats
    const avgEquity = histogram.reduce((sum, prob, i) => sum + prob * (i / (histogram.length - 1)), 0);
    html += '<div class="stats-panel" style="margin-top: 16px;">';
    html += '<div class="stats-grid">';
    html += `<div class="stat-item"><span class="stat-label">Expected Equity</span><span class="stat-value">${(avgEquity * 100).toFixed(1)}%</span></div>`;
    html += `<div class="stat-item"><span class="stat-label">Bins</span><span class="stat-value">${histogram.length}</span></div>`;
    html += '</div>';
    html += '</div>';

    el.viewHistogram.innerHTML = html;
}

// === Initialize ===
initialize();
