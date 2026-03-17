# 🎬 Text-to-Video & Image-to-Video — Quick Cheat Sheet

> **One line:** Type a prompt or drop an image → AI generates a video. No camera, no crew.

---

## 🗺️ Two Technologies at a Glance

```
Text  ──────────→ [Text-to-Video]  ──→ 🎥 Video
Image ──────────→ [Image-to-Video] ──→ 🎥 Animated Video
```

---

## 📝 Text-to-Video — How It Works

```
1. Text Encoding
   Prompt → Language Model → High-dimensional semantic vector

2. Latent Space Generation
   Semantic vector → Diffusion Model → Sequence of latent frames
   (starts from random noise → iteratively refined)

3. Temporal Consistency  ← Key challenge
   3D U-Nets   → capture spatial + time dynamics
   Transformers → self-attention across frames (smooth motion)

4. Video Decoding
   Latent frames → CNN Decoder → Actual video frames

5. Frame Interpolation (Optional)
   Fill gaps between keyframes → smoother, higher frame rate
```

---

## 🖼️ Image-to-Video — How It Works

```
1. Feature Extraction
   Image → CNN → edges, textures, semantic content

2. Motion Prediction
   Optical Flow  → pixel-level motion between frames
   Latent Flow   → motion in compressed space (faster, coherent)

3. Frame Generation
   GANs → realistic frames (generator vs. discriminator)
   VAEs → diverse, coherent frame distribution

4. Video Assembly
   Frames compiled → stabilization + color correction → final video
```

---

## 🏆 Notable Models (2024–2025)

### Text-to-Video

| Model | Key Strength | Access |
|---|---|---|
| **OpenAI Sora** | 60s videos, complex scenes, camera motion | ChatGPT Plus/Pro |
| **Google Veo 2** | Cinematic quality, physics modeling | Gemini Advanced |
| **Runway Gen-4** | Consistent characters, storytelling control | Paid/Enterprise |
| **Step-Video-T2V** | 30B params, 204-frame long videos | Open-source |
| **AMD Hummingbird** | Lightweight, 31× speedup, only 4 GPUs | Open-source |

### Image-to-Video

| Model | Key Strength | Access |
|---|---|---|
| **OpenAI Sora** | Realistic motion + scene transitions | ChatGPT Plus/Pro |
| **Google Whisk Animate** | 8s, 720p animated videos | Google One AI Premium |
| **I2V3D** | 3D camera movement, geometry-aware | Open-source |
| **MiniMax Hailuo I2V** | High motion control from single image | Hailuo AI platform |

---

## 🌍 Real-World Applications

| Industry | Use Case |
|---|---|
| **Marketing** | Promo videos, product showcases, localized ads at scale |
| **Education** | Animated tutorials, multilingual learning content |
| **Entertainment** | Storyboards, VFX, music video visuals |
| **Social Media** | Short-form video, GIFs, viral content |
| **Corporate** | Onboarding videos, policy explainers |

---

## ⚠️ Challenges

| Area | Problem |
|---|---|
| **Compute** | Huge GPU requirements; hard for small orgs |
| **Coherence** | Flickering, unnatural transitions, inconsistent lighting |
| **Data** | Needs large, high-quality, annotated video datasets |
| **Ethics** | Deepfakes, misinformation, copyright/consent issues |
| **Control** | Hard to steer exact output; limited fine-grained control |

---

## 🔭 Future Directions

```
→ Efficient architectures   (edge devices, real-time)
→ Better prompt control     (finer output direction)
→ Multimodal input          (text + image + audio → video)
→ Ethical frameworks        (watermarking, content verification)
→ Personalized video        (adaptive storytelling per viewer)
```

---

## ✅ Key Takeaways

```
1. Text-to-Video  = Diffusion model turns prompt → coherent video
2. Image-to-Video = Motion prediction animates a static image
3. Temporal consistency = the hardest technical problem
4. Already live in marketing, education, entertainment
5. Ethical guardrails still catching up to the technology
```

---

*Source: cognitiveclass.ai — Introduction to Text-to-Video and Image-to-Video Technologies*
