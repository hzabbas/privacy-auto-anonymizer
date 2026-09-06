/**
 * Privacy Auto-Anonymizer - Visual Rendering & Canvas Interaction
 * Modern HTML5 Canvas rendering layer with neon bounding boxes and AI inference.
 */

class AnonymizerStudio {
    constructor() {
        // DOM Elements
        this.canvas = document.getElementById('editor-canvas') || document.getElementById('imageCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.dropzone = document.getElementById('dropzoneContainer');
        this.placeholder = document.getElementById('placeholderState');
        this.uploadInputs = document.querySelectorAll('input[type="file"]');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.statusBadge = document.getElementById('statusBadge');
        this.statusText = document.getElementById('statusText');
        this.legendContainer = document.getElementById('legendContainer');
        
        // Control Buttons
        this.btnSelectAll = document.getElementById('btnSelectAll');
        this.btnDeselectAll = document.getElementById('btnDeselectAll');
        this.btnExport = document.getElementById('btnExport');

        // State
        this.originalImage = null;
        this.currentFile = null;
        this.detections = []; // [{ id, type, bbox: [x1, y1, x2, y2], score, active: true }]
        this.isProcessing = false;

        // Color themes for entities
        this.colorConfig = {
            face: {
                stroke: '#06b6d4',      // Neon Cyan
                fill: 'rgba(6, 182, 212, 0.15)',
                badgeBg: 'rgba(6, 182, 212, 0.90)',
                badgeText: '#ffffff',
                label: 'Face'
            },
            plate: {
                stroke: '#10b981',     // Neon Emerald
                fill: 'rgba(16, 185, 129, 0.15)',
                badgeBg: 'rgba(16, 185, 129, 0.90)',
                badgeText: '#ffffff',
                label: 'Plate'
            },
            text: {
                stroke: '#f59e0b',      // Neon Amber
                fill: 'rgba(245, 158, 11, 0.15)',
                badgeBg: 'rgba(245, 158, 11, 0.90)',
                badgeText: '#ffffff',
                label: 'Text'
            },
            default: {
                stroke: '#8b5cf6',      // Purple fallback
                fill: 'rgba(139, 92, 246, 0.15)',
                badgeBg: 'rgba(139, 92, 246, 0.90)',
                badgeText: '#ffffff',
                label: 'Entity'
            }
        };

        this.initEventListeners();
    }

    initEventListeners() {
        // File inputs (header and dropzone)
        this.uploadInputs.forEach(input => {
            input.addEventListener('change', (e) => {
                const file = e.target.files && e.target.files[0];
                if (file) this.loadFile(file);
            });
        });

        // Drag & Drop
        if (this.dropzone) {
            ['dragenter', 'dragover'].forEach(eventName => {
                this.dropzone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    this.dropzone.classList.add('border-zinc-500', 'bg-zinc-900/40');
                });
            });

            ['dragleave', 'drop'].forEach(eventName => {
                this.dropzone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    this.dropzone.classList.remove('border-zinc-500', 'bg-zinc-900/40');
                });
            });

            this.dropzone.addEventListener('drop', (e) => {
                const dt = e.dataTransfer;
                if (dt && dt.files && dt.files[0]) {
                    this.loadFile(dt.files[0]);
                }
            });
        }

        // Selection buttons
        if (this.btnSelectAll) {
            this.btnSelectAll.addEventListener('click', () => {
                this.detections.forEach(d => d.active = true);
                this.render();
            });
        }

        if (this.btnDeselectAll) {
            this.btnDeselectAll.addEventListener('click', () => {
                this.detections.forEach(d => d.active = false);
                this.render();
            });
        }
    }

    loadFile(file) {
        if (!file || !file.type.startsWith('image/')) {
            alert('لطفاً یک فایل تصویری معتبر (JPG, PNG, WEBP) انتخاب کنید.');
            return;
        }

        this.currentFile = file;
        this.detections = [];

        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                this.originalImage = img;
                
                // Initialize canvas dimensions to exact native image resolution
                this.canvas.width = img.naturalWidth;
                this.canvas.height = img.naturalHeight;

                // Adjust UI visibility
                if (this.placeholder) this.placeholder.classList.add('hidden');
                this.canvas.classList.remove('hidden');
                if (this.dropzone) {
                    this.dropzone.classList.remove('border-dashed');
                    this.dropzone.classList.add('border-solid');
                }

                // Initial render of raw image
                this.render();

                // Trigger AI inference API call
                this.analyzeImage(file);
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }

    async analyzeImage(file) {
        this.setLoading(true);
        if (this.statusBadge) this.statusBadge.classList.add('hidden');

        try {
            const formData = new FormData();
            formData.append('image', file);
            formData.append('conf_threshold', '0.35');

            const response = await fetch('/api/analyze/', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.message || `خطای سرور: ${response.status}`);
            }

            const data = await response.json();
            if (data.status === 'success' && Array.isArray(data.detections)) {
                // Store detections with default active state = true
                this.detections = data.detections.map(det => ({
                    ...det,
                    active: true
                }));

                // Enable toolbar controls
                this.enableControls(true);

                // Update status indicator
                this.updateStatusSummary();

                // Re-render canvas with bounding boxes
                this.render();
            } else {
                throw new Error(data.message || 'پاسخ نامعتبر از سرور دریافت شد.');
            }
        } catch (error) {
            console.error('[Anonymizer] Error during image analysis:', error);
            alert(`خطا در تحلیل تصویر: ${error.message}`);
        } finally {
            this.setLoading(false);
        }
    }

    setLoading(loading) {
        this.isProcessing = loading;
        if (this.loadingOverlay) {
            if (loading) {
                this.loadingOverlay.classList.remove('hidden');
                this.loadingOverlay.classList.add('flex');
            } else {
                this.loadingOverlay.classList.remove('flex');
                this.loadingOverlay.classList.add('hidden');
            }
        }
    }

    enableControls(enabled) {
        [this.btnSelectAll, this.btnDeselectAll, this.btnExport].forEach(btn => {
            if (btn) {
                btn.disabled = !enabled;
            }
        });
        if (this.legendContainer) {
            if (enabled && this.detections.length > 0) {
                this.legendContainer.classList.remove('hidden');
                this.legendContainer.classList.add('flex');
            }
        }
    }

    updateStatusSummary() {
        if (!this.statusBadge || !this.statusText) return;

        const count = this.detections.length;
        if (count === 0) {
            this.statusText.textContent = 'هیچ عنصر حساسی شناسایی نشد';
        } else {
            const faces = this.detections.filter(d => d.type === 'face').length;
            const plates = this.detections.filter(d => d.type === 'plate').length;
            const texts = this.detections.filter(d => d.type === 'text').length;

            const parts = [];
            if (faces > 0) parts.push(`${faces} چهره`);
            if (plates > 0) parts.push(`${plates} پلاک`);
            if (texts > 0) parts.push(`${texts} متن`);

            this.statusText.textContent = `${count} مورد حساس یافت شد (${parts.join('، ')})`;
        }

        this.statusBadge.classList.remove('hidden');
        this.statusBadge.classList.add('flex');
    }

    render() {
        if (!this.originalImage) return;

        const { width, height } = this.canvas;
        const ctx = this.ctx;

        // 1. Draw original base image
        ctx.clearRect(0, 0, width, height);
        ctx.drawImage(this.originalImage, 0, 0, width, height);

        // 2. Draw suggestion bounding boxes
        this.detections.forEach(det => {
            this.drawDetectionBox(ctx, det);
        });
    }

    drawDetectionBox(ctx, det) {
        const [x1, y1, x2, y2] = det.bbox;
        const boxWidth = x2 - x1;
        const boxHeight = y2 - y1;

        if (boxWidth <= 0 || boxHeight <= 0) return;

        const config = this.colorConfig[det.type] || this.colorConfig.default;
        const isActive = det.active !== false;

        ctx.save();

        if (isActive) {
            // Neon box fill
            ctx.fillStyle = config.fill;
            ctx.fillRect(x1, y1, boxWidth, boxHeight);

            // Neon stroke
            ctx.strokeStyle = config.stroke;
            ctx.lineWidth = 2.5;
            ctx.shadowColor = config.stroke;
            ctx.shadowBlur = 6;
            ctx.strokeRect(x1, y1, boxWidth, boxHeight);

            // Subtle corner accents
            this.drawCornerAccents(ctx, x1, y1, boxWidth, boxHeight, config.stroke);

            // Pill label
            this.drawBadgeLabel(ctx, x1, y1, det, config);
        } else {
            // Inactive / deselected state (muted dashed outline)
            ctx.strokeStyle = 'rgba(113, 113, 122, 0.5)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 4]);
            ctx.strokeRect(x1, y1, boxWidth, boxHeight);
        }

        ctx.restore();
    }

    drawCornerAccents(ctx, x, y, w, h, color) {
        const length = Math.min(10, w / 4, h / 4);
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.shadowBlur = 0;

        // Top-left
        ctx.beginPath();
        ctx.moveTo(x, y + length);
        ctx.lineTo(x, y);
        ctx.lineTo(x + length, y);
        ctx.stroke();

        // Top-right
        ctx.beginPath();
        ctx.moveTo(x + w - length, y);
        ctx.lineTo(x + w, y);
        ctx.lineTo(x + w, y + length);
        ctx.stroke();

        // Bottom-left
        ctx.beginPath();
        ctx.moveTo(x, y + h - length);
        ctx.lineTo(x, y + h);
        ctx.lineTo(x + length, y + h);
        ctx.stroke();

        // Bottom-right
        ctx.beginPath();
        ctx.moveTo(x + w - length, y + h);
        ctx.lineTo(x + w, y + h);
        ctx.lineTo(x + w, y + h - length);
        ctx.stroke();

        ctx.restore();
    }

    drawBadgeLabel(ctx, x, y, det, config) {
        const scorePercent = Math.round(det.score * 100);
        const labelText = `${config.label} ${scorePercent}%`;

        // Responsive font size relative to image resolution
        const fontSize = Math.max(12, Math.min(16, Math.round(this.canvas.width / 60)));
        ctx.font = `600 ${fontSize}px Inter, sans-serif`;

        const paddingX = 8;
        const paddingY = 4;
        const textMetrics = ctx.measureText(labelText);
        const badgeWidth = textMetrics.width + (paddingX * 2);
        const badgeHeight = fontSize + (paddingY * 2);

        // Position label above box if space permits, else inside top
        let badgeY = y - badgeHeight - 4;
        if (badgeY < 4) {
            badgeY = y + 4;
        }

        let badgeX = x;
        if (badgeX + badgeWidth > this.canvas.width - 4) {
            badgeX = this.canvas.width - badgeWidth - 4;
        }

        // Draw pill container
        ctx.fillStyle = config.badgeBg;
        ctx.shadowColor = 'rgba(0, 0, 0, 0.4)';
        ctx.shadowBlur = 4;
        this.roundRect(ctx, badgeX, badgeY, badgeWidth, badgeHeight, 4);
        ctx.fill();

        // Draw pill text
        ctx.fillStyle = config.badgeText;
        ctx.shadowBlur = 0;
        ctx.textBaseline = 'middle';
        ctx.fillText(labelText, badgeX + paddingX, badgeY + (badgeHeight / 2));
    }

    roundRect(ctx, x, y, width, height, radius) {
        if (ctx.roundRect) {
            ctx.beginPath();
            ctx.roundRect(x, y, width, height, radius);
            return;
        }
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + width - radius, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        ctx.lineTo(x + width, y + height - radius);
        ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        ctx.lineTo(x + radius, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.closePath();
    }
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    window.anonymizerStudio = new AnonymizerStudio();
});
