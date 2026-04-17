/**
 * app.js — Employee Attrition AI
 * Prediction form + Navigation only (dashboard uses Looker Studio)
 */

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initPredictionForm();
});

// ============================================
// NAVIGATION
// ============================================
function initNavigation() {
    const links = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.section');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                links.forEach(l => l.classList.remove('active'));
                const match = document.querySelector(`.nav-link[data-section="${entry.target.id}"]`);
                if (match) match.classList.add('active');
            }
        });
    }, { rootMargin: '-40% 0px -50% 0px' });

    sections.forEach(s => observer.observe(s));
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
        loadingText.style.display = 'inline-flex';
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
            console.error('Prediction error:', err);
            alert('Gagal: ' + err.message);
        } finally {
            defaultText.style.display = 'inline-flex';
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

    box.className = 'result-display ' + (isSafe ? 'safe' : 'danger');
    emoji.textContent = isSafe ? '✅' : '⚠️';
    status.textContent = result.label;
    desc.textContent = isSafe
        ? 'Karyawan ini diprediksi akan tetap bertahan di perusahaan.'
        : 'Karyawan ini beresiko resign. Perlu perhatian terhadap faktor-faktor terkait.';

    setTimeout(() => {
        document.getElementById('pctBertahan').textContent = result.probability.bertahan + '%';
        document.getElementById('pctResign').textContent = result.probability.resign + '%';
        document.getElementById('barBertahan').style.width = result.probability.bertahan + '%';
        document.getElementById('barResign').style.width = result.probability.resign + '%';
    }, 100);

    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
