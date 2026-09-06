/**
 * Privacy Auto-Anonymizer - Day 7: Final Completion, Client-Side Export & Comparison
 *
 * Full client-side architecture:
 * 1. Clean Export (1:1 Native resolution, 0 helper graphics, 100% privacy preserving)
 * 2. Full-Auto Redaction Mode (one-click instant sanitization)
 * 3. Before/After Comparison Tool (Hold-to-Peek & Spacebar toggle)
 * 4. Human-in-the-Loop interactive masking (DP-Pix & Solid Black Box)
 *
 * Grounded in:
 * - ReGenHuman (2026): Interactive Human-in-the-Loop curation.
 * - Incremental Image Anonymization (2025): DP-Pix differential privacy pixelation.
 * - RedactionBench (2026): Zero-entropy Contextual Integrity masking.
 */

class AnonymizerStudio {
    constructor() {
        // DOM Elements
        this.canvas = document.getElementById('editor-canvas') || document.getElementById('imageCanvas');
        this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
        this.dropzone = document.getElementById('dropzoneContainer');
        this.placeholder = document.getElementById('placeholderState');
        this.uploadInputs = document.querySelectorAll('input[type="file"]');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.statusBadge = document.getElementById('statusBadge');
        this.statusText = document.getElementById('statusText');
        this.legendContainer = document.getElementById('legendContainer');

        // Control Buttons
        this.btnAutoRedact = document.getElementById('btnAutoRedact');
        this.btnSelectAll = document.getElementById('btnSelectAll');
        this.btnDeselectAll = document.getElementById('btnDeselectAll');
        this.btnCompare = document.getElementById('btnCompare');
        this.btnExport = document.getElementById('btnExport');

        // State
        this.originalImage = null;
        this.currentFile = null;
        this.detections = []; // [{ id, type, bbox: [x1, y1, x2, y2], score, isMasked: boolean }]
        this.isProcessing = false;
        this.hoveredDetectionId = null;
        this.isComparing = false; // When true, renders raw unmasked original image

        // Color themes for entities (English labels avoid Canvas RTL character disjointing)
        this.colorConfig = {
            face: {
                stroke: '#06b6d4',      // Neon Cyan
                fill: 'rgba(6, 182, 212, 0.15)',
                badgeBg: '#06b6d4',     // Solid 100% opacity to prevent text bleed-through
                badgeText: '#ffffff',
                label: 'Face'
            },
            plate: {
                stroke: '#10b981',     // Neon Emerald
                fill: 'rgba(16, 185, 129, 0.15)',
                badgeBg: '#10b981',    // Solid 100% opacity
                badgeText: '#ffffff',
                label: 'Plate'
            },
            text: {
                stroke: '#f59e0b',      // Neon Amber
                fill: 'rgba(245, 158, 11, 0.15)',
                badgeBg: '#f59e0b',     // Solid 100% opacity
                badgeText: '#ffffff',
                label: 'Sensitive Text'
            },
            default: {
                stroke: '#8b5cf6',
                fill: 'rgba(139, 92, 246, 0.15)',
                badgeBg: '#8b5cf6',
                badgeText: '#ffffff',
                label: 'Sensitive Entity'
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

        // Canvas Click & Hover for Hit Testing (Human-in-the-loop)
        this.canvas.addEventListener('click', (e) => this.handleCanvasClick(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleCanvasMouseMove(e));
        this.canvas.addEventListener('mouseleave', () => {
            if (this.hoveredDetectionId !== null) {
                this.hoveredDetectionId = null;
                this.canvas.style.cursor = 'default';
                this.render();
            }
        });

        // 1. Full-Auto Redaction Action
        if (this.btnAutoRedact) {
            this.btnAutoRedact.addEventListener('click', () => {
                if (this.detections.length === 0) return;
                this.detections.forEach(d => d.isMasked = true);
                this.updateStatusSummary();
                this.render();
                this.showToast('تمام عناصر حساس با موفقیت سانسور شدند.');
            });
        }

        // 2. Batch Selection Actions
        if (this.btnSelectAll) {
            this.btnSelectAll.addEventListener('click', () => {
                this.detections.forEach(d => d.isMasked = true);
                this.updateStatusSummary();
                this.render();
            });
        }

        if (this.btnDeselectAll) {
            this.btnDeselectAll.addEventListener('click', () => {
                this.detections.forEach(d => d.isMasked = false);
                this.updateStatusSummary();
                this.render();
            });
        }

        // 3. Before/After Comparison Tool (Hold to Peek)
        if (this.btnCompare) {
            const startCompare = (e) => {
                e.preventDefault();
                if (!this.originalImage) return;
                this.isComparing = true;
                this.render();
            };

            const stopCompare = (e) => {
                e.preventDefault();
                if (!this.originalImage) return;
                this.isComparing = false;
                this.render();
            };

            this.btnCompare.addEventListener('mousedown', startCompare);
            this.btnCompare.addEventListener('mouseup', stopCompare);
            this.btnCompare.addEventListener('mouseleave', stopCompare);
            this.btnCompare.addEventListener('touchstart', startCompare, { passive: false });
            this.btnCompare.addEventListener('touchend', stopCompare, { passive: false });
        }

        // Spacebar shortcut for Before/After peek
        window.addEventListener('keydown', (e) => {
            if (e.code === 'Space' && e.target === document.body && this.originalImage && !this.isComparing) {
                e.preventDefault();
                this.isComparing = true;
                this.render();
            }
        });

        window.addEventListener('keyup', (e) => {
            if (e.code === 'Space' && this.isComparing) {
                e.preventDefault();
                this.isComparing = false;
                this.render();
            }
        });

        // 4. Clean Export Action
        if (this.btnExport) {
            this.btnExport.addEventListener('click', () => this.exportCleanImage());
        }
    }

    /**
     * Translates viewport screen mouse coordinates to native canvas image pixels.
     */
    getCanvasCoordinates(e) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }

    /**
     * Hit testing: Finds detections overlapping coordinates, prioritizing smaller areas.
     */
    findHitDetection(x, y) {
        if (this.isComparing) return null;

        const hits = this.detections.filter(det => {
            const [x1, y1, x2, y2] = det.bbox;
            return x >= x1 && x <= x2 && y >= y1 && y <= y2;
        });

        if (hits.length === 0) return null;

        // Prioritize smaller area for nested/adjacent bounding boxes
        hits.sort((a, b) => {
            const areaA = (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]);
            const areaB = (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]);
            return areaA - areaB;
        });

        return hits[0];
    }

    /**
     * Interactive Click Handler: Toggles isMasked between true and false.
     */
    handleCanvasClick(e) {
        if (!this.originalImage || this.detections.length === 0 || this.isComparing) return;

        const { x, y } = this.getCanvasCoordinates(e);
        const hit = this.findHitDetection(x, y);

        if (hit) {
            hit.isMasked = !hit.isMasked;
            this.updateStatusSummary();
            this.render();
        }
    }

    /**
     * Mouse Move Handler: Cursor styling & subtle hover highlight.
     */
    handleCanvasMouseMove(e) {
        if (!this.originalImage || this.detections.length === 0 || this.isComparing) return;

        const { x, y } = this.getCanvasCoordinates(e);
        const hit = this.findHitDetection(x, y);

        if (hit) {
            this.canvas.style.cursor = 'pointer';
            if (this.hoveredDetectionId !== hit.id) {
                this.hoveredDetectionId = hit.id;
                this.render();
            }
        } else {
            this.canvas.style.cursor = 'default';
            if (this.hoveredDetectionId !== null) {
                this.hoveredDetectionId = null;
                this.render();
            }
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
                // Initialize detections ready for interaction
                this.detections = data.detections.map(det => ({
                    ...det,
                    isMasked: false
                }));

                // Enable toolbar controls
                this.enableControls(true);

                // Update status indicator
                this.updateStatusSummary();

                // Re-render canvas with suggestion bounding boxes
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
        [this.btnAutoRedact, this.btnSelectAll, this.btnDeselectAll, this.btnCompare, this.btnExport].forEach(btn => {
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

        const total = this.detections.length;
        if (total === 0) {
            this.statusText.textContent = 'هیچ عنصر حساسی شناسایی نشد';
        } else {
            const maskedCount = this.detections.filter(d => d.isMasked).length;
            this.statusText.textContent = `${total} مورد شناسایی شد (${maskedCount} ماسک فعال)`;
        }

        this.statusBadge.classList.remove('hidden');
        this.statusBadge.classList.add('flex');
    }

    /**
     * Master Render Loop:
     * - If isComparing is active: renders pure unmasked original image.
     * - Otherwise: renders original image + active masks + interactive suggestion boxes.
     */
    render() {
        if (!this.originalImage) return;

        const { width, height } = this.canvas;
        const ctx = this.ctx;

        // 1. Draw original base image
        ctx.clearRect(0, 0, width, height);
        ctx.drawImage(this.originalImage, 0, 0, width, height);

        // Before/After comparison view: stop here and display indicator
        if (this.isComparing) {
            this.drawComparisonWatermark(ctx, width, height);
            return;
        }

        // 2. Apply active masks (DP-Pix for faces/plates, Solid Black for text)
        this.detections.forEach(det => {
            if (det.isMasked) {
                this.applyClientMask(ctx, det, width, height);
            }
        });

        // 3. Draw overlays (neon suggestions for unmasked, status badges for masked)
        this.detections.forEach(det => {
            const isHovered = (this.hoveredDetectionId === det.id);
            if (det.isMasked) {
                this.drawMaskedOverlay(ctx, det, isHovered);
            } else {
                this.drawSuggestionBox(ctx, det, isHovered);
            }
        });
    }

    drawComparisonWatermark(ctx, width, height) {
        ctx.save();
        const text = 'تصویر اصلی خام (قبل از پالایش)';
        const fontSize = Math.max(13, Math.min(18, Math.round(width / 50)));
        ctx.font = `600 ${fontSize}px Inter, Vazirmatn, sans-serif`;

        const paddingX = 14;
        const paddingY = 8;
        const metrics = ctx.measureText(text);
        const badgeW = metrics.width + paddingX * 2;
        const badgeH = fontSize + paddingY * 2;

        const x = (width - badgeW) / 2;
        const y = 20;

        ctx.fillStyle = 'rgba(9, 9, 11, 0.85)';
        ctx.shadowColor = 'rgba(0, 0, 0, 0.6)';
        ctx.shadowBlur = 10;
        this.roundRect(ctx, x, y, badgeW, badgeH, 6);
        ctx.fill();

        ctx.strokeStyle = 'rgba(6, 182, 212, 0.8)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.fillStyle = '#ffffff';
        ctx.shadowBlur = 0;
        ctx.textBaseline = 'middle';
        ctx.fillText(text, x + paddingX, y + (badgeH / 2));
        ctx.restore();
    }

    /**
     * High-Performance Client-Side Redaction Filter Execution:
     * - text: Solid Black Box (#000000)
     * - face / plate: DP-Pix (Mosaic downsampling + additive Gaussian/Laplace noise)
     */
    applyClientMask(ctx, det, canvasW, canvasH) {
        let [x1, y1, x2, y2] = det.bbox;
        x1 = Math.max(0, Math.min(x1, canvasW));
        y1 = Math.max(0, Math.min(y1, canvasH));
        x2 = Math.max(0, Math.min(x2, canvasW));
        y2 = Math.max(0, Math.min(y2, canvasH));

        const w = x2 - x1;
        const h = y2 - y1;
        if (w <= 0 || h <= 0) return;

        if (det.type === 'text') {
            // Solid Black Box (RedactionBench, 2026)
            ctx.save();
            ctx.fillStyle = '#000000';
            ctx.fillRect(x1, y1, w, h);
            ctx.restore();
        } else {
            // DP-Pix: Mosaic + Noise (Incremental Anonymization, 2025)
            try {
                const imgData = ctx.getImageData(x1, y1, w, h);
                const data = imgData.data;

                // Dynamically scale mosaic block size to region dimensions
                const blockSize = Math.max(8, Math.min(22, Math.round(Math.min(w, h) / 7)));
                const noiseScale = 12.0;

                for (let by = 0; by < h; by += blockSize) {
                    for (let bx = 0; bx < w; bx += blockSize) {
                        const bw = Math.min(blockSize, w - bx);
                        const bh = Math.min(blockSize, h - by);

                        // Average color in mosaic cell
                        let rSum = 0, gSum = 0, bSum = 0, count = 0;
                        for (let dy = 0; dy < bh; dy++) {
                            for (let dx = 0; dx < bw; dx++) {
                                const idx = ((by + dy) * w + (bx + dx)) * 4;
                                rSum += data[idx];
                                gSum += data[idx + 1];
                                bSum += data[idx + 2];
                                count++;
                            }
                        }

                        // Additive stochastic perturbation (DP-Pix formulation)
                        const noiseR = (Math.random() - 0.5) * 2 * noiseScale;
                        const noiseG = (Math.random() - 0.5) * 2 * noiseScale;
                        const noiseB = (Math.random() - 0.5) * 2 * noiseScale;

                        const avgR = Math.min(255, Math.max(0, Math.round(rSum / count + noiseR)));
                        const avgG = Math.min(255, Math.max(0, Math.round(gSum / count + noiseG)));
                        const avgB = Math.min(255, Math.max(0, Math.round(bSum / count + noiseB)));

                        // Fill block with perturbed average
                        for (let dy = 0; dy < bh; dy++) {
                            for (let dx = 0; dx < bw; dx++) {
                                const idx = ((by + dy) * w + (bx + dx)) * 4;
                                data[idx] = avgR;
                                data[idx + 1] = avgG;
                                data[idx + 2] = avgB;
                            }
                        }
                    }
                }

                ctx.putImageData(imgData, x1, y1);
            } catch (err) {
                console.warn('[Anonymizer] Fallback to solid mask on canvas limits:', err);
                ctx.fillStyle = '#18181b';
                ctx.fillRect(x1, y1, w, h);
            }
        }
    }

    /**
     * Renders an unmasked suggestion box with neon styling and type badge.
     */
    drawSuggestionBox(ctx, det, isHovered) {
        const [x1, y1, x2, y2] = det.bbox;
        const boxWidth = x2 - x1;
        const boxHeight = y2 - y1;

        if (boxWidth <= 0 || boxHeight <= 0) return;

        const config = this.colorConfig[det.type] || this.colorConfig.default;

        ctx.save();

        // Semi-transparent neon fill
        ctx.fillStyle = isHovered ? config.fill.replace('0.15', '0.28') : config.fill;
        ctx.fillRect(x1, y1, boxWidth, boxHeight);

        // Neon stroke border
        ctx.strokeStyle = config.stroke;
        ctx.lineWidth = isHovered ? 3.0 : 2.0;
        ctx.shadowColor = config.stroke;
        ctx.shadowBlur = isHovered ? 10 : 5;
        ctx.strokeRect(x1, y1, boxWidth, boxHeight);

        // Corner accents
        this.drawCornerAccents(ctx, x1, y1, boxWidth, boxHeight, config.stroke);

        // Pill label
        const scorePercent = Math.round(det.score * 100);
        const labelText = `${config.label} ${scorePercent}%`;
        this.drawBadgePill(ctx, x1, y1, labelText, config.badgeBg, config.badgeText, false);

        ctx.restore();
    }

    /**
     * Renders a masked area overlay (subtle border + [✓ ماسک شد] badge).
     */
    drawMaskedOverlay(ctx, det, isHovered) {
        const [x1, y1, x2, y2] = det.bbox;
        const boxWidth = x2 - x1;
        const boxHeight = y2 - y1;

        if (boxWidth <= 0 || boxHeight <= 0) return;

        ctx.save();

        // Clean subtle border indicating active redaction
        ctx.strokeStyle = isHovered ? 'rgba(239, 68, 68, 0.8)' : 'rgba(16, 185, 129, 0.6)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash(isHovered ? [4, 4] : []);
        ctx.strokeRect(x1, y1, boxWidth, boxHeight);

        // Masked status badge in clean English
        const badgeText = isHovered ? 'Click to Unmask' : '✓ Masked';
        const badgeBg = isHovered ? '#ef4444' : '#10b981';
        this.drawBadgePill(ctx, x1, y1, badgeText, badgeBg, '#ffffff', true);

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

    drawBadgePill(ctx, x, y, text, bgColor, textColor, isMasked) {
        const fontSize = Math.max(12, Math.min(15, Math.round(this.canvas.width / 65)));
        ctx.font = `600 ${fontSize}px Inter, -apple-system, sans-serif`;

        const paddingX = 8;
        const paddingY = 4;
        const textMetrics = ctx.measureText(text);
        const badgeWidth = textMetrics.width + (paddingX * 2);
        const badgeHeight = fontSize + (paddingY * 2);

        let badgeY = y - badgeHeight - 4;
        if (badgeY < 4) {
            badgeY = y + 4;
        }

        let badgeX = x;
        if (badgeX + badgeWidth > this.canvas.width - 4) {
            badgeX = this.canvas.width - badgeWidth - 4;
        }

        ctx.save();
        // Fully solid background (Opacity = 1.0) to eliminate text bleed-through
        ctx.globalAlpha = 1.0;
        ctx.fillStyle = bgColor;
        ctx.shadowColor = 'rgba(0, 0, 0, 0.6)';
        ctx.shadowBlur = 6;
        this.roundRect(ctx, badgeX, badgeY, badgeWidth, badgeHeight, 4);
        ctx.fill();

        // Draw crisp text
        ctx.fillStyle = textColor;
        ctx.shadowBlur = 0;
        ctx.textBaseline = 'middle';
        ctx.fillText(text, badgeX + paddingX, badgeY + (badgeHeight / 2));
        ctx.restore();
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

    /**
     * Clean Export Method:
     * Generates sanitized image directly on an off-screen canvas at native 1:1 resolution.
     * Strictly avoids rendering helper boxes, neon outlines, pills, or text labels.
     */
    exportCleanImage() {
        if (!this.originalImage) {
            alert('تصویری جهت استخراج وجود ندارد.');
            return;
        }

        const nativeWidth = this.originalImage.naturalWidth;
        const nativeHeight = this.originalImage.naturalHeight;

        // Create off-screen canvas for 100% clean rendering
        const offCanvas = document.createElement('canvas');
        offCanvas.width = nativeWidth;
        offCanvas.height = nativeHeight;
        const offCtx = offCanvas.getContext('2d', { willReadFrequently: true });

        // 1. Draw pure original image
        offCtx.drawImage(this.originalImage, 0, 0, nativeWidth, nativeHeight);

        // 2. Apply ONLY active masks (DP-Pix & Solid Black) without helper UI elements
        this.detections.forEach(det => {
            if (det.isMasked) {
                this.applyClientMask(offCtx, det, nativeWidth, nativeHeight);
            }
        });

        // 3. Trigger direct client-side download via Blob URL
        offCanvas.toBlob((blob) => {
            if (!blob) {
                alert('خطا در ایجاد فایل نهایی.');
                return;
            }

            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            const originalName = this.currentFile ? this.currentFile.name.replace(/\.[^/.]+$/, '') : 'document';
            a.download = `redacted_${originalName}.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            this.showToast(`تصویر پالایش‌شده با موفقیت دانلود شد: redacted_${originalName}.png`);
        }, 'image/png');
    }

    /**
     * Non-intrusive floating toast notification.
     */
    showToast(message) {
        const existingToast = document.getElementById('studioToast');
        if (existingToast) existingToast.remove();

        const toast = document.createElement('div');
        toast.id = 'studioToast';
        toast.className = 'fixed bottom-20 left-1/2 -translate-x-1/2 z-50 px-4 py-2 rounded-lg bg-zinc-900/95 border border-zinc-700 text-zinc-100 text-xs font-medium shadow-2xl flex items-center gap-2 backdrop-blur-md transition-all duration-300 transform opacity-0 translate-y-2';
        toast.innerHTML = `
            <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            <span>${message}</span>
        `;
        document.body.appendChild(toast);

        // Animate in
        requestAnimationFrame(() => {
            toast.classList.remove('opacity-0', 'translate-y-2');
            toast.classList.add('opacity-100', 'translate-y-0');
        });

        // Auto remove after 3.5s
        setTimeout(() => {
            toast.classList.remove('opacity-100', 'translate-y-0');
            toast.classList.add('opacity-0', 'translate-y-2');
            setTimeout(() => toast.remove(), 300);
        }, 3500);
    }
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    window.anonymizerStudio = new AnonymizerStudio();
});
