/**
 * app.js - Frontend Logic
 */

// ============================================
// NAV TAB ACTIVE STATE
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    const tabs = document.querySelectorAll('.tab');
    const sections = document.querySelectorAll('.content-section');

    // Intersection Observer to highlight active tab on scroll
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                tabs.forEach(t => t.classList.remove('active'));
                const match = document.querySelector(`.tab[data-section="${entry.target.id}"]`);
                if (match) match.classList.add('active');
            }
        });
    }, { rootMargin: '-40% 0px -50% 0px' });

    sections.forEach(s => observer.observe(s));

    // Init form
    initPredictionForm();
    loadSavedDashboard();
});

// ============================================
// DASHBOARD EMBED
// ============================================
function loadDashboard() {
    const input = document.getElementById('lookerUrl');
    let url = input.value.trim();
    if (!url) {
        input.focus();
        return;
    }

    // Auto-fix URL if not embed
    if (url.includes('/reporting/') && !url.includes('/embed/')) {
        url = url.replace('/reporting/', '/embed/reporting/');
    }

    const container = document.getElementById('dashboardContent');
    const emptyState = document.getElementById('emptyState');

    emptyState.style.display = 'none';

    const iframe = document.createElement('iframe');
    iframe.src = url;
    iframe.setAttribute('allowfullscreen', '');
    container.appendChild(iframe);

    localStorage.setItem('lookerUrl', url);
}

function loadSavedDashboard() {
    const saved = localStorage.getItem('lookerUrl');
    if (saved) {
        const input = document.getElementById('lookerUrl');
        if (input) {
            input.value = saved;
            loadDashboard();
        }
    }
}

// ============================================
// PREDICTION FORM
// ============================================
function initPredictionForm() {
    const form = document.getElementById('predictionForm');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const btn = document.getElementById('predictBtn');
        const defaultText = document.getElementById('btnDefault');
        const loadingText = document.getElementById('btnLoading');

        defaultText.style.display = 'none';
        loadingText.style.display = 'inline';
        btn.disabled = true;

        const formData = new FormData(form);
        const data = {};
        for (const [key, value] of formData.entries()) {
            data[key] = value;
        }

        try {
            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await res.json();

            if (result.error) {
                alert('Error: ' + result.error);
                return;
            }

            showResult(result);

        } catch (err) {
            alert('Gagal menghubungi server. Pastikan Flask berjalan.');
        } finally {
            defaultText.style.display = 'inline';
            loadingText.style.display = 'none';
            btn.disabled = false;
        }
    });
}

// ============================================
// SHOW RESULT
// ============================================
function showResult(result) {
    const card = document.getElementById('resultCard');
    const box = document.getElementById('resultBox');
    const emoji = document.getElementById('resultEmoji');
    const status = document.getElementById('resultStatus');
    const desc = document.getElementById('resultDesc');

    card.style.display = 'block';

    const isSafe = result.prediction === 0;

    box.className = 'result-box ' + (isSafe ? 'safe' : 'danger');
    emoji.textContent = isSafe ? '✅' : '⚠️';
    status.textContent = result.label;
    desc.textContent = isSafe
        ? 'Berdasarkan data yang diberikan, karyawan ini diprediksi akan tetap bertahan di perusahaan.'
        : 'Karyawan ini diprediksi beresiko resign. Perlu perhatian lebih terhadap faktor-faktor terkait.';

    // Animate bars
    setTimeout(() => {
        document.getElementById('pctBertahan').textContent = result.probability.bertahan + '%';
        document.getElementById('pctResign').textContent = result.probability.resign + '%';
        document.getElementById('barBertahan').style.width = result.probability.bertahan + '%';
        document.getElementById('barResign').style.width = result.probability.resign + '%';
    }, 100);

    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
