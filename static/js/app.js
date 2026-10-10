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
        this.canvasContainer = document.getElementById('canvasContainer');
        this.connectorLayer = document.getElementById('connector-layer');
        this.labelsLayer = document.getElementById('labels-layer');
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

        // Color themes for entities
        this.colorConfig = {
            face: {
                stroke: '#06b6d4',      // Neon Cyan
                fill: 'rgba(6, 182, 212, 0.15)',
                label: 'چهره (Face)'
            },
            plate: {
                stroke: '#10b981',     // Neon Emerald
                fill: 'rgba(16, 185, 129, 0.15)',
                label: 'پلاک خودرو (Plate)'
            },
            text: {
                stroke: '#f59e0b',      // Neon Amber
                fill: 'rgba(245, 158, 11, 0.15)',
                label: 'متن حساس (Text)'
            },
            default: {
                stroke: '#8b5cf6',
                fill: 'rgba(139, 92, 246, 0.15)',
                label: 'عنصر حساس'
            }
        };

        this.initEventListeners();
    }

    escapeHtml(value) {
        return String(value).replace(/[&<>"']/g, (c) => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[c]));
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

        // Engine select change listeners for Auto-Reanalyze
        const ocrSelect = document.getElementById('ocrEngineSelect');
        const layoutSelect = document.getElementById('layoutEngineSelect');
        const imajevToggle = document.getElementById('imajevToggle');
        const reanalyzeHandler = () => {
            if (this.currentFile) {
                this.analyzeImage(this.currentFile);
            }
        };
        if (ocrSelect) ocrSelect.addEventListener('change', reanalyzeHandler);
        if (layoutSelect) layoutSelect.addEventListener('change', reanalyzeHandler);

        if (imajevToggle) {
            imajevToggle.addEventListener('change', (e) => {
                this.handleImajevToggle(e.target.checked);
                reanalyzeHandler();
            });
        }

        const btnClearLogs = document.getElementById('btnClearImajevLogs');
        if (btnClearLogs) {
            btnClearLogs.addEventListener('click', () => {
                const content = document.getElementById('imajevLogContent');
                const countBadge = document.getElementById('imajevLogCount');
                if (content) {
                    content.innerHTML = `<div class="text-zinc-500 text-[11px] italic p-2.5 bg-zinc-900/30 rounded-lg border border-zinc-800/40">لاگ‌ها پاکسازی شدند.</div>`;
                }
                if (countBadge) countBadge.textContent = '0 رویداد';
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

        // Responsive HUD Callouts on Window Resize
        window.addEventListener('resize', () => {
            if (this.originalImage) {
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
    isPointInPolygon(point, polygon) {
        let inside = false;
        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
            const xi = polygon[i][0], yi = polygon[i][1];
            const xj = polygon[j][0], yj = polygon[j][1];
            const intersect = ((yi > point.y) !== (yj > point.y)) &&
                (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi);
            if (intersect) inside = !inside;
        }
        return inside;
    }

    /**
     * Hit testing: Finds detections overlapping coordinates, prioritizing smaller areas.
     */
    findHitDetection(x, y) {
        if (this.isComparing) return null;

        const hits = this.detections.filter(det => {
            if (det.polygon && Array.isArray(det.polygon) && det.polygon.length >= 3) {
                return this.isPointInPolygon({ x, y }, det.polygon);
            }
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
                if (this.canvasContainer) this.canvasContainer.classList.remove('hidden');
                this.canvas.classList.remove('hidden');
                if (this.dropzone) {
                    this.dropzone.classList.remove('border-dashed', 'overflow-hidden');
                    this.dropzone.classList.add('border-solid', 'overflow-visible');
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
        if (this.isProcessing) {
            console.warn("[Anonymizer] Already processing an image. Please wait.");
            return;
        }
        
        this.setLoading(true);
        if (this.statusBadge) this.statusBadge.classList.add('hidden');

        try {
            const formData = new FormData();
            formData.append('image', file);
            formData.append('conf_threshold', '0.35');
            
            const ocrEngine = document.getElementById('ocrEngineSelect')?.value || 'easyocr';
            const layoutEngine = document.getElementById('layoutEngineSelect')?.value || 'regex';
            formData.append('ocr_engine', ocrEngine);
            formData.append('layout_engine', layoutEngine);

            const imajevToggle = document.getElementById('imajevToggle');
            const useImajev = imajevToggle ? imajevToggle.checked : false;
            formData.append('use_imajev', useImajev ? 'true' : 'false');

            if (useImajev) {
                const statusText = document.getElementById('imajevStatusText');
                const statusPulse = document.getElementById('imajevStatusPulse');
                if (statusText) statusText.textContent = 'در حال استدلال...';
                if (statusPulse) {
                    statusPulse.classList.remove('bg-emerald-400');
                    statusPulse.classList.add('bg-cyan-400');
                }
                const container = document.getElementById('imajevLogContent');
                if (container) {
                    container.innerHTML = `<div class="text-cyan-400 text-[11px] font-mono flex items-center gap-2 p-2.5 bg-zinc-900/50 rounded-lg border border-zinc-800/60 animate-pulse">
                        <span class="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
                        در حال ارسال تصویر و اجرای استدلال معنایی توسط Imajev VLM...
                    </div>`;
                }
            }

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

                // Handle Imajev logs display
                if (useImajev && Array.isArray(data.imajev_logs)) {
                    this.renderImajevLogs(data.imajev_logs);
                } else if (!useImajev) {
                    const statusText = document.getElementById('imajevStatusText');
                    if (statusText) statusText.textContent = 'غیرفعال';
                }

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
            this.detections = [];
            this.clearCalloutLayers();
            this.render();
            if (this.statusBadge) this.statusBadge.classList.add('hidden');
            if (typeof this.showToast === 'function') {
                this.showToast(`خطا: ${error.message}`, true);
            } else {
                alert(`خطا در تحلیل تصویر: ${error.message}`);
            }
        } finally {
            this.setLoading(false);
        }
    }

    showToast(message, isError = false) {
        let toast = document.getElementById('app-toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'app-toast';
            toast.className = 'fixed bottom-5 right-5 px-4 py-3 rounded-lg shadow-xl text-sm font-medium z-[100] transition-all duration-300 translate-y-20 opacity-0';
            document.body.appendChild(toast);
        }
        
        toast.className = `fixed bottom-5 right-5 px-4 py-3 rounded-lg shadow-xl text-sm font-medium z-[100] transition-all duration-300 transform translate-y-0 opacity-100 ${isError ? 'bg-red-500/90 text-white' : 'bg-zinc-800 text-emerald-400'}`;
        toast.textContent = message;
        
        if (this.toastTimeout) clearTimeout(this.toastTimeout);
        this.toastTimeout = setTimeout(() => {
            toast.classList.replace('translate-y-0', 'translate-y-20');
            toast.classList.replace('opacity-100', 'opacity-0');
        }, 4000);
    }

    setLoading(loading) {
        this.isProcessing = loading;
        if (this.loadingOverlay) {
            const textElement = this.loadingOverlay.querySelector('p');
            
            if (loading) {
                this.loadingOverlay.classList.remove('hidden');
                this.loadingOverlay.classList.add('flex');
                
                if (textElement) {
                    textElement.style.transition = 'opacity 0.3s ease';
                    textElement.style.opacity = '1';
                    textElement.textContent = "در حال پردازش تصاویر...";
                    
                    if (this.loadingTimeout) clearTimeout(this.loadingTimeout);
                    
                    this.loadingTimeout = setTimeout(() => {
                        if (this.isProcessing) {
                            textElement.style.opacity = '0';
                            setTimeout(() => {
                                textElement.textContent = "در حال بارگذاری یا دانلود مدل‌های هوشمند (در اولین اجرا ممکن است چند دقیقه زمان ببرد)...";
                                textElement.style.opacity = '1';
                            }, 300);
                        }
                    }, 6000);
                }
            } else {
                this.loadingOverlay.classList.remove('flex');
                this.loadingOverlay.classList.add('hidden');
                
                if (this.loadingTimeout) {
                    clearTimeout(this.loadingTimeout);
                    this.loadingTimeout = null;
                }
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
        const dot = this.statusBadge.querySelector('span.w-2');
        
        if (total === 0) {
            this.statusText.textContent = 'هیچ عنصر حساسی یافت نشد';
            if (dot) {
                dot.classList.remove('bg-emerald-500');
                dot.classList.add('bg-amber-500');
            }
        } else {
            const maskedCount = this.detections.filter(d => d.isMasked).length;
            this.statusText.textContent = `${total} مورد شناسایی شد (${maskedCount} ماسک فعال)`;
            if (dot) {
                dot.classList.remove('bg-amber-500');
                dot.classList.add('bg-emerald-500');
            }
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

        // Before/After comparison view: stop here, clear HUD callouts, and display watermark
        if (this.isComparing) {
            this.clearCalloutLayers();
            this.drawComparisonWatermark(ctx, width, height);
            return;
        }

        // 2. Apply active masks (DP-Pix for faces/plates, Solid Black for text)
        this.detections.forEach(det => {
            if (det.isMasked) {
                this.applyClientMask(ctx, det, width, height);
            }
        });

        // 3. Draw overlays on canvas (ONLY strokes and corner accents, NO text)
        this.detections.forEach(det => {
            const isHovered = (this.hoveredDetectionId === det.id);
            if (det.isMasked) {
                this.drawMaskedOverlay(ctx, det, isHovered);
            } else {
                this.drawSuggestionBox(ctx, det, isHovered);
            }
        });

        // 4. Render modern HUD Callouts (SVG connector lines + HTML interactive badges)
        this.renderHUDCallouts();
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
     * - text: Solid Black Mask (#000000) adhering strictly to tilted / oriented polygon
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
            // Solid Black Mask adhering strictly to the tilted/oriented polygon
            ctx.save();
            ctx.fillStyle = '#000000';
            if (det.polygon && Array.isArray(det.polygon) && det.polygon.length >= 3) {
                ctx.beginPath();
                ctx.moveTo(det.polygon[0][0], det.polygon[0][1]);
                for (let i = 1; i < det.polygon.length; i++) {
                    ctx.lineTo(det.polygon[i][0], det.polygon[i][1]);
                }
                ctx.closePath();
                ctx.fill();
            } else {
                ctx.fillRect(x1, y1, w, h);
            }
            ctx.restore();
        } else {
            // DP-Pix: Mosaic + Noise (Incremental Anonymization, 2025)
            try {
                ctx.save();
                if (det.polygon && Array.isArray(det.polygon) && det.polygon.length >= 3) {
                    ctx.beginPath();
                    ctx.moveTo(det.polygon[0][0], det.polygon[0][1]);
                    for (let i = 1; i < det.polygon.length; i++) {
                        ctx.lineTo(det.polygon[i][0], det.polygon[i][1]);
                    }
                    ctx.closePath();
                    ctx.clip();
                }

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
                ctx.restore();
            } catch (err) {
                console.warn('[Anonymizer] Fallback to solid mask on canvas limits:', err);
                ctx.fillStyle = '#18181b';
                ctx.fillRect(x1, y1, w, h);
                ctx.restore();
            }
        }
    }

    /**
     * Renders an unmasked suggestion box with neon styling (tilted polygon or box).
     */
    drawSuggestionBox(ctx, det, isHovered) {
        const config = this.colorConfig[det.type] || this.colorConfig.default;
        ctx.save();

        if (det.polygon && Array.isArray(det.polygon) && det.polygon.length >= 3) {
            ctx.beginPath();
            ctx.moveTo(det.polygon[0][0], det.polygon[0][1]);
            for (let i = 1; i < det.polygon.length; i++) {
                ctx.lineTo(det.polygon[i][0], det.polygon[i][1]);
            }
            ctx.closePath();

            // Semi-transparent neon fill
            ctx.fillStyle = isHovered ? config.fill.replace('0.15', '0.28') : config.fill;
            ctx.fill();

            // Neon stroke border along tilted polygon
            ctx.strokeStyle = config.stroke;
            ctx.lineWidth = isHovered ? 3.0 : 2.0;
            ctx.shadowColor = config.stroke;
            ctx.shadowBlur = isHovered ? 10 : 5;
            ctx.stroke();
        } else {
            const [x1, y1, x2, y2] = det.bbox;
            const boxWidth = x2 - x1;
            const boxHeight = y2 - y1;

            if (boxWidth <= 0 || boxHeight <= 0) {
                ctx.restore();
                return;
            }

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
        }

        ctx.restore();
    }

    /**
     * Renders a masked area overlay (subtle border without canvas text).
     */
    drawMaskedOverlay(ctx, det, isHovered) {
        ctx.save();

        // Clean subtle border indicating active redaction
        ctx.strokeStyle = isHovered ? 'rgba(239, 68, 68, 0.9)' : 'rgba(16, 185, 129, 0.7)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash(isHovered ? [4, 4] : []);

        if (det.polygon && Array.isArray(det.polygon) && det.polygon.length >= 3) {
            ctx.beginPath();
            ctx.moveTo(det.polygon[0][0], det.polygon[0][1]);
            for (let i = 1; i < det.polygon.length; i++) {
                ctx.lineTo(det.polygon[i][0], det.polygon[i][1]);
            }
            ctx.closePath();
            ctx.stroke();
        } else {
            const [x1, y1, x2, y2] = det.bbox;
            const boxWidth = x2 - x1;
            const boxHeight = y2 - y1;

            if (boxWidth <= 0 || boxHeight <= 0) {
                ctx.restore();
                return;
            }
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

    /**
     * Clears both SVG connector lines and HTML callout labels.
     */
    clearCalloutLayers() {
        if (this.connectorLayer) {
            this.connectorLayer.innerHTML = '';
        }
        if (this.labelsLayer) {
            this.labelsLayer.innerHTML = '';
        }
    }

    /**
     * Modern HUD Callout Architecture:
     * Renders angled SVG polyline connectors and floating HTML callout badges
     * positioned around the detected boxes. Eliminates Canvas text rendering
     * artifacts and provides flawless RTL Persian text presentation.
     */
    renderHUDCallouts() {
        this.clearCalloutLayers();
        if (!this.connectorLayer || !this.labelsLayer || this.detections.length === 0) return;

        const rect = this.canvas.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;

        const scaleX = rect.width / this.canvas.width;
        const scaleY = rect.height / this.canvas.height;

        // Prepare detection coordinate data mapped to rendered DOM pixels
        const items = this.detections.map(det => {
            const [x1, y1, x2, y2] = det.bbox;
            const sx1 = x1 * scaleX;
            const sy1 = y1 * scaleY;
            const sx2 = x2 * scaleX;
            const sy2 = y2 * scaleY;
            const boxCenterX = (sx1 + sx2) / 2;
            const boxCenterY = (sy1 + sy2) / 2;

            return {
                det,
                sx1, sy1, sx2, sy2,
                boxWidth: sx2 - sx1,
                boxHeight: sy2 - sy1,
                boxCenterX, boxCenterY,
                direction: (boxCenterX < rect.width / 2) ? 'left' : 'right',
                targetY: boxCenterY
            };
        });

        // 1. Divide detections into Left and Right groups based on X coordinate (left half vs right half)
        const leftGroup = items.filter(it => it.direction === 'left');
        const rightGroup = items.filter(it => it.direction === 'right');

        // 2. Sort each group from top to bottom by Y axis
        leftGroup.sort((a, b) => a.boxCenterY - b.boxCenterY);
        rightGroup.sort((a, b) => a.boxCenterY - b.boxCenterY);

        // 3. Smart Stacking: enforce at least 50px spacing and strictly clamp within canvas height
        const canvasH = rect.height;
        const minSpacing = 50;
        const topMargin = 25;
        const bottomMargin = 35;

        [leftGroup, rightGroup].forEach(group => {
            if (group.length === 0) return;

            // Downward spacing pass
            let lastY = topMargin - minSpacing;
            group.forEach(item => {
                let assignedY = Math.max(item.boxCenterY, topMargin);
                if (assignedY < lastY + minSpacing) {
                    assignedY = lastY + minSpacing;
                }
                item.targetY = assignedY;
                lastY = assignedY;
            });

            // Upward clamp pass: strictly ensures bottom-most label never exceeds canvas height or overlaps controls
            const maxBottom = canvasH - bottomMargin;
            for (let i = group.length - 1; i >= 0; i--) {
                const maxAllowedY = maxBottom - (group.length - 1 - i) * 36;
                group[i].targetY = Math.min(group[i].targetY, maxAllowedY);
                group[i].targetY = Math.max(group[i].targetY, topMargin + i * 36);
            }
        });

        // Document fragments for fast atomic DOM insertion
        const svgFragment = document.createDocumentFragment();
        const htmlFragment = document.createDocumentFragment();

        items.forEach(item => {
            const { det, sx1, sy1, sx2, sy2, boxCenterY, direction, targetY } = item;
            const config = this.colorConfig[det.type] || this.colorConfig.default;
            const isHovered = (this.hoveredDetectionId === det.id);
            const isMasked = det.isMasked;
            const strokeColor = isMasked ? (isHovered ? '#ef4444' : '#10b981') : config.stroke;

            // Strict Y clamping: ensure labels never exceed canvas bottom
            const clampedY = Math.min(targetY, canvasH - 35);

            // Adaptive Positioning: check space between canvas and dropzoneContainer bounds
            const dropzoneRect = this.dropzone ? this.dropzone.getBoundingClientRect() : rect;
            const spaceLeft = rect.left - dropzoneRect.left;
            const spaceRight = dropzoneRect.right - rect.right;
            const BADGE_WIDTH = 120;
            const SAFE_PAD = 15;
            const NEEDED_SPACE = BADGE_WIDTH + SAFE_PAD; // 135px

            const canFitOutsideLeft = spaceLeft >= NEEDED_SPACE;
            const canFitOutsideRight = spaceRight >= NEEDED_SPACE;

            // Connector & Badge Geometry:
            let p0, p1, p2, p3;
            let badgeLeft;

            if (direction === 'right') {
                if (canFitOutsideRight) {
                    p0 = { x: sx2, y: boxCenterY };
                    p1 = { x: sx2 + 10, y: boxCenterY };
                    p2 = { x: rect.width + 8, y: clampedY };
                    p3 = { x: rect.width + 18, y: clampedY };
                    badgeLeft = `${(rect.width + 18).toFixed(1)}px`;
                } else if (canFitOutsideLeft) {
                    p0 = { x: sx1, y: boxCenterY };
                    p1 = { x: sx1 - 10, y: boxCenterY };
                    p2 = { x: -8, y: clampedY };
                    p3 = { x: -18, y: clampedY };
                    badgeLeft = `${(-BADGE_WIDTH - 18).toFixed(1)}px`;
                } else {
                    // Dock inside canvas right edge
                    const targetX = rect.width - BADGE_WIDTH - 12;
                    p0 = { x: sx2, y: boxCenterY };
                    p1 = { x: Math.min(targetX - 8, (sx2 + targetX) / 2), y: boxCenterY };
                    p2 = { x: targetX - 6, y: clampedY };
                    p3 = { x: targetX - 2, y: clampedY };
                    badgeLeft = `${targetX.toFixed(1)}px`;
                }
            } else {
                // direction === 'left'
                if (canFitOutsideLeft) {
                    p0 = { x: sx1, y: boxCenterY };
                    p1 = { x: sx1 - 10, y: boxCenterY };
                    p2 = { x: -8, y: clampedY };
                    p3 = { x: -18, y: clampedY };
                    badgeLeft = `${(-BADGE_WIDTH - 18).toFixed(1)}px`;
                } else if (canFitOutsideRight) {
                    p0 = { x: sx2, y: boxCenterY };
                    p1 = { x: sx2 + 10, y: boxCenterY };
                    p2 = { x: rect.width + 8, y: clampedY };
                    p3 = { x: rect.width + 18, y: clampedY };
                    badgeLeft = `${(rect.width + 18).toFixed(1)}px`;
                } else {
                    // Dock inside canvas left edge (safely away from Imajev log panel!)
                    const targetX = 12;
                    p0 = { x: sx1, y: boxCenterY };
                    p1 = { x: Math.max(targetX + BADGE_WIDTH + 8, (sx1 + targetX + BADGE_WIDTH) / 2), y: boxCenterY };
                    p2 = { x: targetX + BADGE_WIDTH + 6, y: clampedY };
                    p3 = { x: targetX + BADGE_WIDTH + 2, y: clampedY };
                    badgeLeft = `${targetX.toFixed(1)}px`;
                }
            }

            // 1. Build SVG Connector Group
            const groupEl = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            groupEl.setAttribute('class', 'hud-connector transition-all duration-200');

            // Anchor dot on detection box edge
            const dot0 = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            dot0.setAttribute('cx', p0.x.toFixed(1));
            dot0.setAttribute('cy', p0.y.toFixed(1));
            dot0.setAttribute('r', isHovered ? '4' : '3');
            dot0.setAttribute('fill', strokeColor);
            groupEl.appendChild(dot0);

            // Polyline connecting box to outer margin
            const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
            polyline.setAttribute('points', `${p0.x.toFixed(1)},${p0.y.toFixed(1)} ${p1.x.toFixed(1)},${p1.y.toFixed(1)} ${p2.x.toFixed(1)},${clampedY.toFixed(1)} ${p3.x.toFixed(1)},${clampedY.toFixed(1)}`);
            polyline.setAttribute('stroke', strokeColor);
            polyline.setAttribute('stroke-width', isHovered ? '2.2' : '1.5');
            polyline.setAttribute('stroke-linecap', 'round');
            polyline.setAttribute('stroke-linejoin', 'round');
            polyline.setAttribute('fill', 'none');
            polyline.setAttribute('opacity', isHovered ? '1' : (isMasked ? '0.75' : '0.9'));
            if (!isMasked && !isHovered) {
                polyline.setAttribute('stroke-dasharray', '4,3');
            }
            groupEl.appendChild(polyline);

            // Terminal dot at label attachment
            const dot3 = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            dot3.setAttribute('cx', p3.x.toFixed(1));
            dot3.setAttribute('cy', clampedY.toFixed(1));
            dot3.setAttribute('r', isHovered ? '3.5' : '2.5');
            dot3.setAttribute('fill', strokeColor);
            groupEl.appendChild(dot3);

            svgFragment.appendChild(groupEl);

            // 2. Build Minimal HTML Callout Badge (Safely positioned)
            const badge = document.createElement('div');
            badge.dataset.detectionId = det.id;

            badge.style.position = 'absolute';
            badge.style.top = `${clampedY.toFixed(1)}px`;
            badge.style.width = `${BADGE_WIDTH}px`;
            badge.style.left = badgeLeft;
            badge.style.transform = 'translate(0, -50%)';

            // High-contrast HUD aesthetics
            const borderCol = isMasked
                ? (isHovered ? 'border-red-500/90 shadow-[0_0_15px_rgba(239,68,68,0.4)]' : 'border-emerald-500/80 shadow-[0_0_12px_rgba(16,185,129,0.3)]')
                : (isHovered ? 'border-cyan-400 shadow-[0_0_15px_rgba(6,182,212,0.4)]' : 'border-zinc-700/90 shadow-[0_4px_12px_rgba(0,0,0,0.6)]');

            const bgCol = isHovered
                ? 'bg-[#141418] text-white scale-[1.04]'
                : 'bg-[#0f0f13]/95 text-zinc-200';

            const scorePercent = Math.round(det.score * 100);

            badge.className = `pointer-events-auto cursor-pointer select-none px-2 py-1.5 rounded-lg border ${borderCol} ${bgCol} text-xs font-medium transition-all duration-150 flex items-center justify-between gap-1.5 backdrop-blur-md whitespace-nowrap z-20 hover:scale-105 active:scale-95 group`;

            // Minimalist content: ONLY entity name and confidence score (no extra button text)
            badge.innerHTML = `
                <div class="flex items-center gap-1.5 overflow-hidden">
                    <span class="w-1.5 h-1.5 rounded-full flex-shrink-0 transition-transform duration-200 group-hover:scale-125" style="background-color: ${strokeColor}; box-shadow: 0 0 6px ${strokeColor};"></span>
                    <span class="text-zinc-100 font-medium tracking-tight text-[10px] truncate">${this.escapeHtml(det.label ? `متن حساس: ${det.label}` : config.label)}</span>
                </div>
                <span class="text-[9px] font-mono px-1 py-0.5 rounded bg-zinc-900/90 text-zinc-300 border border-zinc-700/60 flex-shrink-0">${scorePercent}%</span>
            `;

            // Hover interactions: highlight box, line, and badge
            badge.addEventListener('mouseenter', () => {
                if (this.hoveredDetectionId !== det.id) {
                    this.hoveredDetectionId = det.id;
                    this.render();
                }
            });

            badge.addEventListener('mouseleave', () => {
                if (this.hoveredDetectionId === det.id) {
                    this.hoveredDetectionId = null;
                    this.render();
                }
            });

            // Click interaction: toggle isMasked and re-render
            badge.addEventListener('click', (e) => {
                e.stopPropagation();
                det.isMasked = !det.isMasked;
                this.updateStatusSummary();
                this.render();
            });

            htmlFragment.appendChild(badge);
        });

        this.connectorLayer.appendChild(svgFragment);
        this.labelsLayer.appendChild(htmlFragment);
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
     * Robust client-side file download via dynamically created <a> element with Blob URL.
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

        // 3. Reliable client-side download via Blob
        offCanvas.toBlob((blob) => {
            if (!blob) return;
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `redacted_${Date.now()}.png`;
            document.body.appendChild(a);
            a.click();
            setTimeout(() => {
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
            }, 1000);
            this.showToast('تصویر پالایش‌شده با موفقیت دانلود شد');
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

    /**
     * Toggles 2-column responsive layout shift when Imajev is enabled/disabled.
     */
    handleImajevToggle(isActive) {
        const workspaceGrid = document.getElementById('workspaceGrid');
        const dropzoneContainer = document.getElementById('dropzoneContainer');
        const logPanel = document.getElementById('imajevLogPanel');
        const mainWorkspace = document.getElementById('mainWorkspace');

        if (isActive) {
            if (workspaceGrid) {
                workspaceGrid.classList.remove('grid-cols-1');
                workspaceGrid.classList.add('grid-cols-1', 'lg:grid-cols-4');
            }
            if (dropzoneContainer) {
                dropzoneContainer.classList.add('lg:col-span-3');
            }
            if (logPanel) {
                logPanel.classList.remove('hidden');
                logPanel.classList.add('flex', 'lg:col-span-1');
            }
            if (mainWorkspace) {
                mainWorkspace.classList.remove('max-w-6xl');
                mainWorkspace.classList.add('max-w-7xl');
            }
            this.showToast('حالت استدلال هوشمند Imajev فعال شد.');
        } else {
            if (workspaceGrid) {
                workspaceGrid.classList.remove('lg:grid-cols-4');
                workspaceGrid.classList.add('grid-cols-1');
            }
            if (dropzoneContainer) {
                dropzoneContainer.classList.remove('lg:col-span-3');
            }
            if (logPanel) {
                logPanel.classList.remove('flex', 'lg:col-span-1');
                logPanel.classList.add('hidden');
            }
            if (mainWorkspace) {
                mainWorkspace.classList.remove('max-w-7xl');
                mainWorkspace.classList.add('max-w-6xl');
            }
        }

        // Allow layout animation to settle and refresh canvas HUD alignment
        setTimeout(() => {
            if (this.originalImage) {
                this.render();
            }
        }, 150);
    }

    renderImajevLogs(logs) {
        const container = document.getElementById('imajevLogContent');
        const countBadge = document.getElementById('imajevLogCount');
        const statusText = document.getElementById('imajevStatusText');
        const statusPulse = document.getElementById('imajevStatusPulse');
        if (!container) return;

        container.innerHTML = '';
        if (countBadge) countBadge.textContent = `${logs.length} رویداد`;
        if (statusText) statusText.textContent = 'تکمیل شد';
        if (statusPulse) {
            statusPulse.classList.remove('bg-cyan-400');
            statusPulse.classList.add('bg-emerald-400');
        }

        if (!logs || logs.length === 0) {
            container.innerHTML = `<div class="text-zinc-500 text-[11px] italic p-2.5 bg-zinc-900/30 rounded-lg border border-zinc-800/40">هیچ تصمیمی توسط Imajev برای این تصویر ثبت نشد.</div>`;
            return;
        }

        logs.forEach((logStr, index) => {
            let logData = {};
            try {
                logData = JSON.parse(logStr);
            } catch (e) {
                logData = { entity: 'system', message: logStr, action: 'info' };
            }
            
            const item = document.createElement('div');
            item.className = 'group flex flex-col gap-2 p-3 rounded-xl bg-zinc-900/80 border border-zinc-800/80 hover:border-cyan-500/50 hover:bg-zinc-900 transition-all duration-200 opacity-0 translate-y-2 relative overflow-hidden cursor-pointer shadow-sm';

            if (logData.id) {
                item.dataset.detectionId = logData.id;
                
                // Canvas coordination: hover effects
                item.addEventListener('mouseenter', () => {
                    if (this.hoveredDetectionId !== logData.id) {
                        this.hoveredDetectionId = logData.id;
                        this.render();
                    }
                });
                item.addEventListener('mouseleave', () => {
                    if (this.hoveredDetectionId === logData.id) {
                        this.hoveredDetectionId = null;
                        this.render();
                    }
                });
            }

            let actionBadge = '';
            let barColor = 'bg-cyan-500';
            let barDotColor = 'bg-cyan-400';
            
            if (logData.action === 'approved') {
                actionBadge = `<span class="px-2 py-0.5 rounded text-[10px] font-bold tracking-wider font-mono bg-emerald-950/90 text-emerald-400 border border-emerald-800/80 shadow-[0_0_8px_rgba(16,185,129,0.25)]">[APPROVED]</span>`;
                barColor = 'bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.5)]';
                barDotColor = 'bg-emerald-400';
            } else if (logData.action === 'rejected') {
                actionBadge = `<span class="px-2 py-0.5 rounded text-[10px] font-bold tracking-wider font-mono bg-rose-950/90 text-rose-400 border border-rose-800/80 shadow-[0_0_8px_rgba(244,63,94,0.25)]">[REJECTED]</span>`;
                barColor = 'bg-rose-500 shadow-[0_0_6px_rgba(244,63,94,0.5)]';
                barDotColor = 'bg-rose-400';
            } else if (logData.action === 'error') {
                actionBadge = `<span class="px-2 py-0.5 rounded text-[10px] font-bold tracking-wider font-mono bg-orange-950/90 text-orange-400 border border-orange-800/80">[ERROR]</span>`;
                barColor = 'bg-orange-500';
                barDotColor = 'bg-orange-400';
            } else {
                actionBadge = `<span class="px-2 py-0.5 rounded text-[10px] font-bold tracking-wider font-mono bg-cyan-950/90 text-cyan-400 border border-cyan-800/80 shadow-[0_0_8px_rgba(6,182,212,0.25)]">[INFO]</span>`;
            }

            let entityName = logData.entity ? logData.entity.toUpperCase() : 'SYSTEM';
            if (logData.text_preview) entityName += ` "${logData.text_preview}"`;

            let messageMarkup = '';
            const msg = logData.message || (logData.reason ? `دلیل تصمیم: ${logData.reason}` : '');
            if (msg) {
                messageMarkup = `<div class="text-[11px] text-zinc-300 leading-relaxed font-sans px-0.5" dir="auto">${this.escapeHtml(msg)}</div>`;
            }

            let metricsMarkup = '';
            if (logData.confidence !== undefined) {
                const confPercent = Math.round(logData.confidence * 100);
                const unkPercent = logData.unknown_prob !== undefined ? Math.round(logData.unknown_prob * 100) : 0;
                
                metricsMarkup = `
                    <div class="flex flex-col gap-1.5 pt-2 border-t border-zinc-800/60 text-[10px] font-mono select-none" dir="ltr">
                        <div class="flex items-center gap-2">
                            <span class="text-zinc-400 w-16 flex-shrink-0 flex items-center gap-1">
                                <span class="w-1.5 h-1.5 rounded-full ${barDotColor}"></span>
                                Conf
                            </span>
                            <div class="flex-1 bg-zinc-950 rounded-full h-1.5 overflow-hidden border border-zinc-800/80">
                                <div class="${barColor} h-full rounded-full transition-all duration-500" style="width: ${confPercent}%"></div>
                            </div>
                            <span class="text-zinc-200 font-semibold w-7 text-right">${confPercent}%</span>
                        </div>
                        <div class="flex items-center gap-2">
                            <span class="text-zinc-400 w-16 flex-shrink-0 flex items-center gap-1">
                                <span class="w-1.5 h-1.5 rounded-full bg-amber-400"></span>
                                Unknown
                            </span>
                            <div class="flex-1 bg-zinc-950 rounded-full h-1.5 overflow-hidden border border-zinc-800/80">
                                <div class="bg-amber-400 h-full rounded-full transition-all duration-500" style="width: ${unkPercent}%"></div>
                            </div>
                            <span class="text-amber-400/90 font-semibold w-7 text-right">${unkPercent}%</span>
                        </div>
                    </div>
                `;
            }

            item.innerHTML = `
                <div class="flex items-center justify-between gap-2" dir="ltr">
                    <div class="flex items-center gap-1.5 overflow-hidden">
                        ${actionBadge}
                        <span class="text-[10px] font-semibold text-zinc-300 bg-zinc-800/90 border border-zinc-700/60 px-1.5 py-0.5 rounded tracking-wider uppercase truncate">${this.escapeHtml(entityName)}</span>
                    </div>
                    <span class="text-[9px] font-mono text-zinc-500 bg-zinc-950/60 px-1.5 py-0.5 rounded border border-zinc-800 flex-shrink-0">#${String(index + 1).padStart(2, '0')}</span>
                </div>
                ${messageMarkup}
                ${metricsMarkup}
            `;
            container.appendChild(item);

            setTimeout(() => {
                item.classList.remove('opacity-0', 'translate-y-2');
                item.classList.add('opacity-100', 'translate-y-0');
                container.scrollTop = container.scrollHeight;
            }, index * 80);
        });
    }

    escapeHtml(str) {
        if (!str) return '';
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    window.anonymizerStudio = new AnonymizerStudio();
});
