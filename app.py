"""Streamlit frontend for APUBT3-NUP image compression."""
from __future__ import annotations

import base64
import io

import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="APUBT3-NUP Image Compression",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar ---
with st.sidebar:
    st.title("APUBT3-NUP")
    st.caption("Non-Uniform Partitioning + U-System Transform Coding")

    st.markdown("---")
    st.markdown(
        """
**Based on:**
*"A New Image Compression Algorithm Based on Non-Uniform Partition and U-System"*
IEEE Transactions on Multimedia, 2021

**Pipeline:**
1. ITANRP texture classification (16×16 tiles)
2. Non-uniform block layout (8×8 or 16×16)
3. APUBT3 U-system transform
4. Adaptive scalar quantization
5. Zigzag scan + RLE encoding
        """
    )

    st.markdown("---")
    st.subheader("Backend Status")
    try:
        r = requests.get(f"{API_URL}/", timeout=3)
        if r.ok:
            st.success("Connected to API")
        else:
            st.error("API returned an error")
    except Exception:
        st.error(
            "API not reachable.\n\nStart it with:\n```\nuvicorn backend:app --reload\n```"
        )

    st.markdown("---")
    st.caption("Upload only JPEG (.jpg / .jpeg) images for encode / export / analyze.")


# --- Helper ---
def b64_to_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 ** 2:.2f} MB"


def show_metric_row(meta: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("Original", fmt_bytes(meta.get("original_bytes", 0)))
    cols[1].metric("Compressed", fmt_bytes(meta.get("compressed_bytes", 0)))
    cols[2].metric("PSNR", f"{meta.get('psnr', 0):.2f} dB")
    cols[3].metric("SSIM", f"{meta.get('ssim', 0):.4f}")

    cols2 = st.columns(4)
    cols2[0].metric("Compression Ratio", f"{meta.get('compression_ratio', 0):.3f}")
    cols2[1].metric("BPP", f"{meta.get('bpp', 0):.4f}")
    cols2[2].metric("Target Reached", "Yes" if meta.get("target_reached") else "No")
    cols2[3].metric("Detail Tiles", str(meta.get("detail_tiles", "—")))


# --- Tabs ---
tab_encode, tab_export, tab_decode, tab_analyze = st.tabs(
    ["Encode (.apubt3)", "Export JPEG", "Decode (.apubt3)", "Bitrate Analysis"]
)

# ===========================================================================
# TAB 1 — ENCODE
# ===========================================================================
with tab_encode:
    st.header("Encode to APUBT3-NUP Bitstream")
    st.markdown(
        "Compresses your JPEG into a custom `.apubt3` archive storing the "
        "quantized APUBT3-NUP coefficients. The archive can be decoded back "
        "to a viewable image without any further JPEG loss."
    )

    left, right = st.columns([1, 2], gap="large")

    with left:
        enc_file = st.file_uploader("Upload JPEG", type=["jpg", "jpeg"], key="enc_up")
        enc_quality = st.slider("Quality", 1, 100, 85, key="enc_q",
                                help="Higher = less transform quantization pressure")
        enc_ratio = st.slider(
            "Target ratio (compressed / original)", 0.05, 0.95, 0.60, 0.05,
            key="enc_r",
            help="0.5 means ~half the original file size"
        )
        enc_residual = st.checkbox(
            "Preserve residual correction",
            key="enc_res",
            help="Stores an RGB correction layer to reduce color drift",
        )
        enc_btn = st.button("Encode", type="primary", key="enc_go",
                            disabled=enc_file is None)

    with right:
        if enc_file:
            if enc_btn:
                with st.spinner("Encoding — this may take a few seconds…"):
                    resp = requests.post(
                        f"{API_URL}/api/encode",
                        files={"file": (enc_file.name, enc_file.getvalue(), "image/jpeg")},
                        data={
                            "quality": enc_quality,
                            "target_ratio": enc_ratio,
                            "preserve_residual": str(enc_residual).lower(),
                        },
                        timeout=120,
                    )

                if resp.ok:
                    result = resp.json()
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(enc_file, caption="Original", use_container_width=True)
                    with c2:
                        st.image(
                            b64_to_image(result["reconstructed_b64"]),
                            caption="Reconstructed from .apubt3",
                            use_container_width=True,
                        )

                    st.markdown("#### Metrics")
                    show_metric_row(result)

                    st.download_button(
                        "Download compressed.apubt3",
                        data=base64.b64decode(result["file_b64"]),
                        file_name="compressed.apubt3",
                        mime="application/octet-stream",
                    )

                    with st.expander("Full metadata"):
                        display = {k: v for k, v in result.items()
                                   if k not in ("file_b64", "reconstructed_b64")}
                        st.json(display)
                else:
                    st.error(f"Error {resp.status_code}: {resp.json().get('detail', resp.text)}")
            else:
                st.image(enc_file, caption="Preview — click Encode to compress", use_container_width=True)
        else:
            st.info("Upload a JPEG to get started.")


# ===========================================================================
# TAB 2 — EXPORT JPEG
# ===========================================================================
with tab_export:
    st.header("Export as Compressed JPEG")
    st.markdown(
        "Runs the full APUBT3-NUP pipeline and saves the result directly as a "
        "standard `.jpg` file — convenient for side-by-side viewing and sharing."
    )

    left, right = st.columns([1, 2], gap="large")

    with left:
        exp_file = st.file_uploader("Upload JPEG", type=["jpg", "jpeg"], key="exp_up")
        exp_quality = st.slider("Quality", 1, 100, 85, key="exp_q")
        exp_ratio = st.slider(
            "Target ratio", 0.05, 0.95, 0.60, 0.05, key="exp_r"
        )
        exp_residual = st.checkbox("Preserve residual correction", key="exp_res")
        exp_btn = st.button("Export JPEG", type="primary", key="exp_go",
                            disabled=exp_file is None)

    with right:
        if exp_file:
            if exp_btn:
                with st.spinner("Exporting…"):
                    resp = requests.post(
                        f"{API_URL}/api/export-jpeg",
                        files={"file": (exp_file.name, exp_file.getvalue(), "image/jpeg")},
                        data={
                            "quality": exp_quality,
                            "target_ratio": exp_ratio,
                            "preserve_residual": str(exp_residual).lower(),
                        },
                        timeout=120,
                    )

                if resp.ok:
                    result = resp.json()
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(exp_file, caption="Original", use_container_width=True)
                    with c2:
                        st.image(
                            b64_to_image(result["image_b64"]),
                            caption="APUBT3-NUP compressed JPEG",
                            use_container_width=True,
                        )

                    st.markdown("#### Metrics")
                    show_metric_row(result)

                    st.download_button(
                        "Download compressed.jpg",
                        data=base64.b64decode(result["image_b64"]),
                        file_name="compressed.jpg",
                        mime="image/jpeg",
                    )

                    with st.expander("Full metadata"):
                        display = {k: v for k, v in result.items() if k != "image_b64"}
                        st.json(display)
                else:
                    st.error(f"Error {resp.status_code}: {resp.json().get('detail', resp.text)}")
            else:
                st.image(exp_file, caption="Preview", use_container_width=True)
        else:
            st.info("Upload a JPEG to get started.")


# ===========================================================================
# TAB 3 — DECODE
# ===========================================================================
with tab_decode:
    st.header("Decode APUBT3-NUP Bitstream")
    st.markdown(
        "Reconstruct a viewable image from a `.apubt3` archive produced by the "
        "Encode tab. No additional JPEG loss is introduced during decoding."
    )

    left, right = st.columns([1, 2], gap="large")

    with left:
        dec_file = st.file_uploader("Upload .apubt3 file", type=["apubt3"], key="dec_up")
        dec_btn = st.button("Decode", type="primary", key="dec_go",
                            disabled=dec_file is None)

    with right:
        if dec_file and dec_btn:
            with st.spinner("Decoding…"):
                resp = requests.post(
                    f"{API_URL}/api/decode",
                    files={"file": (dec_file.name, dec_file.getvalue(), "application/octet-stream")},
                    timeout=60,
                )

            if resp.ok:
                result = resp.json()
                img = b64_to_image(result["image_b64"])
                st.image(img, caption="Reconstructed image", use_container_width=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Compressed Size", fmt_bytes(result.get("compressed_bytes", 0)))
                c2.metric("BPP", f"{result.get('bpp', 0):.4f}")
                h, w = result.get("original_shape", [0, 0])
                c3.metric("Image Size", f"{w} × {h}")

                st.download_button(
                    "Download reconstructed.png",
                    data=base64.b64decode(result["image_b64"]),
                    file_name="reconstructed.png",
                    mime="image/png",
                )

                with st.expander("Full metadata"):
                    display = {k: v for k, v in result.items() if k != "image_b64"}
                    st.json(display)
            else:
                st.error(f"Error {resp.status_code}: {resp.json().get('detail', resp.text)}")
        elif not dec_file:
            st.info("Upload a `.apubt3` file produced by the Encode tab.")


# ===========================================================================
# TAB 4 — BITRATE ANALYSIS
# ===========================================================================
with tab_analyze:
    st.header("Bitrate Analysis")
    st.markdown(
        "Evaluates APUBT3-NUP vs baseline JPEG across several compression "
        "ratios and plots PSNR and SSIM vs BPP curves — reproducing the "
        "paper-style quality comparison."
    )

    left, right = st.columns([1, 2], gap="large")

    with left:
        ana_file = st.file_uploader("Upload JPEG", type=["jpg", "jpeg"], key="ana_up")
        ana_quality = st.slider("Quality", 1, 100, 85, key="ana_q")
        ana_ratios = st.multiselect(
            "Target ratios to evaluate",
            options=[0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
            default=[0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
            key="ana_ratios_sel",
        )
        st.caption("More ratios = longer processing time (several seconds each).")
        ana_btn = st.button(
            "Run Analysis", type="primary", key="ana_go",
            disabled=(ana_file is None or len(ana_ratios) == 0),
        )

    with right:
        if ana_file and ana_btn:
            ratios_str = ",".join(str(r) for r in sorted(ana_ratios))
            with st.spinner(f"Running {len(ana_ratios)} ratios × 2 methods — please wait…"):
                resp = requests.post(
                    f"{API_URL}/api/analyze",
                    files={"file": (ana_file.name, ana_file.getvalue(), "image/jpeg")},
                    data={"quality": ana_quality, "ratios": ratios_str},
                    timeout=600,
                )

            if resp.ok:
                rows = resp.json()["rows"]

                # Separate by method
                apubt3_rows = sorted(
                    [r for r in rows if r["method"] == "APUBT3-NUP"], key=lambda r: r["bpp"]
                )
                jpeg_rows = sorted(
                    [r for r in rows if r["method"] == "JPEG"], key=lambda r: r["bpp"]
                )

                # PSNR chart
                fig_psnr = go.Figure()
                fig_psnr.add_trace(go.Scatter(
                    x=[r["bpp"] for r in apubt3_rows],
                    y=[r["psnr"] for r in apubt3_rows],
                    mode="lines+markers", name="APUBT3-NUP",
                    marker=dict(symbol="circle", size=8),
                    line=dict(width=2),
                ))
                fig_psnr.add_trace(go.Scatter(
                    x=[r["bpp"] for r in jpeg_rows],
                    y=[r["psnr"] for r in jpeg_rows],
                    mode="lines+markers", name="JPEG",
                    marker=dict(symbol="square", size=8),
                    line=dict(width=2, dash="dash"),
                ))
                fig_psnr.update_layout(
                    title="PSNR vs BPP",
                    xaxis_title="Bits per pixel (BPP)",
                    yaxis_title="PSNR (dB)",
                    legend=dict(x=0.02, y=0.02),
                    height=380,
                )
                st.plotly_chart(fig_psnr, use_container_width=True)

                # SSIM chart
                fig_ssim = go.Figure()
                fig_ssim.add_trace(go.Scatter(
                    x=[r["bpp"] for r in apubt3_rows],
                    y=[r["ssim"] for r in apubt3_rows],
                    mode="lines+markers", name="APUBT3-NUP",
                    marker=dict(symbol="circle", size=8),
                    line=dict(width=2),
                ))
                fig_ssim.add_trace(go.Scatter(
                    x=[r["bpp"] for r in jpeg_rows],
                    y=[r["ssim"] for r in jpeg_rows],
                    mode="lines+markers", name="JPEG",
                    marker=dict(symbol="square", size=8),
                    line=dict(width=2, dash="dash"),
                ))
                fig_ssim.update_layout(
                    title="SSIM vs BPP",
                    xaxis_title="Bits per pixel (BPP)",
                    yaxis_title="SSIM",
                    legend=dict(x=0.02, y=0.02),
                    height=380,
                )
                st.plotly_chart(fig_ssim, use_container_width=True)

                # Data table
                with st.expander("Raw data table"):
                    import pandas as pd
                    df = pd.DataFrame(rows)
                    df["compressed_bytes"] = df["compressed_bytes"].apply(fmt_bytes)
                    df["original_bytes"] = df["original_bytes"].apply(fmt_bytes)
                    st.dataframe(df, use_container_width=True)
            else:
                st.error(f"Error {resp.status_code}: {resp.json().get('detail', resp.text)}")
        elif not ana_file:
            st.info("Upload a JPEG and click **Run Analysis** to generate bitrate curves.")
