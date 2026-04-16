import * as THREE from 'three';
import { SceneManager } from './SceneManager.js';
import { UIManager } from './UIManager.js';
import { PointCloudManager } from '../managers/PointCloudManager.js';

export class App {
    constructor() {
        this.container = document.getElementById('container');
        this.mouse = new THREE.Vector2();
        this.currentFileName = '';

        this.sceneManager = new SceneManager(this.container);
        this.ui = new UIManager();
        this.pointCloudManager = new PointCloudManager(this.sceneManager.scene);

        this.bindEvents();
        this.animate();
    }

    bindEvents() {
        this.ui.bindFileChange(async (file) => {
            if (!file) return;

            this.currentFileName = file.name;
            this.ui.setLoading(true, '正在解析 PLY 文件...');

            try {
                await this.pointCloudManager.loadPLY(file, this.ui.getPointSize());
                this.pointCloudManager.fitCamera(
                    this.sceneManager.camera,
                    this.sceneManager.controls
                );
                this.ui.setStats(this.pointCloudManager.getStats(this.currentFileName));
            } catch (error) {
                console.error('加载 PLY 失败:', error);
                this.ui.setError(`加载失败：${error.message}`);
            } finally {
                this.ui.setLoading(false);
            }
        });

        this.ui.bindPointSizeChange((size) => {
            this.pointCloudManager.setPointSize(size);
        });

        this.ui.bindBackgroundChange((mode) => {
            this.sceneManager.setBackground(mode);
        });

        this.ui.bindFit(() => {
            this.pointCloudManager.fitCamera(
                this.sceneManager.camera,
                this.sceneManager.controls
            );
        });

        this.ui.bindClear(() => {
            this.pointCloudManager.clear();
            this.currentFileName = '';
            this.ui.setDefaultStats();
        });

        this.ui.bindToggleGrid(() => {
            this.sceneManager.toggleGrid();
        });

        this.ui.bindToggleAxes(() => {
            this.sceneManager.toggleAxes();
        });

        window.addEventListener('resize', () => {
            this.sceneManager.resize();
        });

        this.sceneManager.renderer.domElement.addEventListener('pointerdown', (event) => {
            this.onPointerDown(event);
        });
    }

    onPointerDown(event) {
        if (event.button !== 0) return;
        if (!this.pointCloudManager.pointCloud) return;

        const rect = this.sceneManager.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        const hit = this.pointCloudManager.pick(
            this.mouse,
            this.sceneManager.camera,
            this.ui.getPointSize()
        );

        if (!hit) return;

        const point = hit.point.clone();
        this.pointCloudManager.showPickMarker(point);

        this.ui.setStats(`
      ${this.pointCloudManager.getStats(this.currentFileName)}<br />
      拾取点：(${point.x.toFixed(5)}, ${point.y.toFixed(5)}, ${point.z.toFixed(5)})
    `);
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.sceneManager.render();
    }
}
