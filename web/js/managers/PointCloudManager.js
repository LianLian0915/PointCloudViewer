import * as THREE from 'three';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';

export class PointCloudManager {
    constructor(scene) {
        this.scene = scene;
        this.pointCloud = null;
        this.pickMarker = null;
        this.raycaster = new THREE.Raycaster();
        this.lastLoadInfo = null;
    }

    removeNaNVertices(geometry) {
        const posAttr = geometry.getAttribute('position');
        if (!posAttr) {
            throw new Error('PLY 文件中未找到 position 顶点属性');
        }

        const positions = posAttr.array;
        const colors = geometry.getAttribute('color')?.array || null;
        const normals = geometry.getAttribute('normal')?.array || null;

        const validPositions = [];
        const validColors = [];
        const validNormals = [];

        for (let i = 0; i < positions.length; i += 3) {
            const x = positions[i];
            const y = positions[i + 1];
            const z = positions[i + 2];

            if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
                continue;
            }

            validPositions.push(x, y, z);

            if (colors && colors.length >= i + 3) {
                validColors.push(colors[i], colors[i + 1], colors[i + 2]);
            }

            if (normals && normals.length >= i + 3) {
                validNormals.push(normals[i], normals[i + 1], normals[i + 2]);
            }
        }

        const cleanGeometry = new THREE.BufferGeometry();
        cleanGeometry.setAttribute('position', new THREE.Float32BufferAttribute(validPositions, 3));

        if (validColors.length > 0) {
            cleanGeometry.setAttribute('color', new THREE.Float32BufferAttribute(validColors, 3));
        }

        if (validNormals.length > 0) {
            cleanGeometry.setAttribute('normal', new THREE.Float32BufferAttribute(validNormals, 3));
        }

        return cleanGeometry;
    }

    ensureVertexColors(geometry) {
        const pos = geometry.getAttribute('position');
        let colorAttr = geometry.getAttribute('color');

        if (!colorAttr) {
            const count = pos.count;
            const colors = new Float32Array(count * 3);

            for (let i = 0; i < count; i++) {
                colors[i * 3] = 0.15;
                colors[i * 3 + 1] = 0.8;
                colors[i * 3 + 2] = 1.0;
            }

            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            colorAttr = geometry.getAttribute('color');
        }

        const arr = colorAttr.array;
        let maxVal = 0;

        for (let i = 0; i < Math.min(arr.length, 3000); i++) {
            if (arr[i] > maxVal) maxVal = arr[i];
        }

        if (maxVal > 1.0) {
            for (let i = 0; i < arr.length; i++) {
                arr[i] = arr[i] / 255.0;
            }
            colorAttr.needsUpdate = true;
        }
    }

    escapeHTML(value) {
        return String(value).replace(/[&<>"']/g, (char) => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;'
        })[char]);
    }

    async loadPLY(file, pointSize = 0.01) {
        const t0 = performance.now();

        const arrayBuffer = await file.arrayBuffer();
        const t1 = performance.now();

        const loader = new PLYLoader();
        const parsedGeometry = loader.parse(arrayBuffer);
        const t2 = performance.now();

        const geometry = this.removeNaNVertices(parsedGeometry);
        this.ensureVertexColors(geometry);

        geometry.computeBoundingBox();
        geometry.computeBoundingSphere();

        if (!geometry.boundingSphere || !Number.isFinite(geometry.boundingSphere.radius)) {
            throw new Error('点云包围球计算失败，可能是 PLY 数据异常');
        }

        this.clear(false);

        const material = new THREE.PointsMaterial({
            size: pointSize,
            vertexColors: true,
            sizeAttenuation: true
        });

        this.pointCloud = new THREE.Points(geometry, material);
        this.scene.add(this.pointCloud);

        const t3 = performance.now();

        this.lastLoadInfo = {
            fileName: file.name,
            totalMs: t3 - t0,
            readMs: t1 - t0,
            parseMs: t2 - t1,
            buildMs: t3 - t2
        };

        return this.pointCloud;
    }

    setPointSize(size) {
        if (this.pointCloud?.material) {
            this.pointCloud.material.size = size;
            this.pointCloud.material.needsUpdate = true;
        }
    }

    getStats(fileName = '') {
        if (!this.pointCloud) {
            return '尚未加载点云';
        }

        const geometry = this.pointCloud.geometry;
        const count = geometry.getAttribute('position')?.count || 0;
        const box = geometry.boundingBox;
        const sphere = geometry.boundingSphere;

        const size = new THREE.Vector3();
        box.getSize(size);

        const loadInfo = this.lastLoadInfo;
        const loadHtml = loadInfo
            ? `
        加载耗时：${loadInfo.totalMs.toFixed(2)} ms<br />
        文件读取：${loadInfo.readMs.toFixed(2)} ms<br />
        PLY解析：${loadInfo.parseMs.toFixed(2)} ms<br />
        点云构建：${loadInfo.buildMs.toFixed(2)} ms<br />
      `
            : '';

        return `
文件：${this.escapeHTML(fileName)}<br />
点数：${count.toLocaleString()}<br />
包围盒尺寸：${size.x.toFixed(4)} × ${size.y.toFixed(4)} × ${size.z.toFixed(4)}<br />
包围球半径：${sphere.radius.toFixed(4)}<br />
${loadHtml}
    `.trim();
    }

    fitCamera(camera, controls) {
        if (!this.pointCloud) return;

        const geometry = this.pointCloud.geometry;
        if (!geometry.boundingSphere) {
            geometry.computeBoundingSphere();
        }

        const sphere = geometry.boundingSphere;
        const center = sphere.center.clone();
        const radius = Math.max(sphere.radius, 0.001);

        controls.target.copy(center);

        const fov = camera.fov * (Math.PI / 180);
        const distance = radius / Math.sin(fov / 2);
        const dir = new THREE.Vector3(1, 0.8, 1).normalize();

        camera.position.copy(center.clone().add(dir.multiplyScalar(distance * 1.2)));
        camera.near = Math.max(radius / 1000, 0.001);
        camera.far = Math.max(radius * 20, 1000);
        camera.updateProjectionMatrix();
        controls.update();
    }

    pick(mouseNdc, camera, pointSize = 0.01) {
        if (!this.pointCloud) return null;

        this.raycaster.params.Points.threshold = pointSize * 1.5 + 0.01;
        this.raycaster.setFromCamera(mouseNdc, camera);

        const intersects = this.raycaster.intersectObject(this.pointCloud);
        if (intersects.length === 0) {
            return null;
        }

        return intersects[0];
    }

    showPickMarker(point) {
        this.clearPickMarker();

        this.pickMarker = new THREE.Mesh(
            new THREE.SphereGeometry(0.01, 16, 16),
            new THREE.MeshBasicMaterial({ color: 0xff3b30 })
        );

        this.pickMarker.position.copy(point);
        this.scene.add(this.pickMarker);
    }

    clearPickMarker() {
        if (this.pickMarker) {
            this.scene.remove(this.pickMarker);
            this.pickMarker.geometry.dispose();
            this.pickMarker.material.dispose();
            this.pickMarker = null;
        }
    }

    clear(resetMarker = true) {
        if (this.pointCloud) {
            this.scene.remove(this.pointCloud);
            this.pointCloud.geometry.dispose();
            this.pointCloud.material.dispose();
            this.pointCloud = null;
        }

        if (resetMarker) {
            this.clearPickMarker();
        }

        this.lastLoadInfo = null;
    }
}
