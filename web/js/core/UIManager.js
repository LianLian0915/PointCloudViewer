export class UIManager {
    constructor() {
        this.fileInput = document.getElementById('fileInput');
        this.pointSizeInput = document.getElementById('pointSize');
        this.pointSizeValue = document.getElementById('pointSizeValue');
        this.bgMode = document.getElementById('bgMode');
        this.fitBtn = document.getElementById('fitBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.toggleGridBtn = document.getElementById('toggleGridBtn');
        this.toggleAxesBtn = document.getElementById('toggleAxesBtn');
        this.statsEl = document.getElementById('stats');
        this.loadingEl = document.getElementById('loading');
    }

    bindFileChange(handler) {
        this.fileInput.addEventListener('change', (e) => {
            const file = e.target.files?.[0] || null;
            handler(file);
        });
    }

    bindPointSizeChange(handler) {
        this.pointSizeInput.addEventListener('input', () => {
            const value = Number(this.pointSizeInput.value);
            this.pointSizeValue.textContent = value.toFixed(3);
            handler(value);
        });
    }

    bindBackgroundChange(handler) {
        this.bgMode.addEventListener('change', () => {
            handler(this.bgMode.value);
        });
    }

    bindFit(handler) {
        this.fitBtn.addEventListener('click', handler);
    }

    bindClear(handler) {
        this.clearBtn.addEventListener('click', handler);
    }

    bindToggleGrid(handler) {
        this.toggleGridBtn.addEventListener('click', handler);
    }

    bindToggleAxes(handler) {
        this.toggleAxesBtn.addEventListener('click', handler);
    }

    setLoading(show, text = '正在加载点云...') {
        this.loadingEl.style.display = show ? 'block' : 'none';
        this.loadingEl.textContent = text;
    }

    setStats(html) {
        this.statsEl.innerHTML = html;
    }

    setDefaultStats() {
        this.statsEl.textContent = '尚未加载点云';
    }

    getPointSize() {
        return Number(this.pointSizeInput.value);
    }
}