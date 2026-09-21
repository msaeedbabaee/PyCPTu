import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


class CPTuEngine:

  def __init__(self, p_atm=101.325, gamma_w=9.81):
    self.p_atm = p_atm
    self.gamma_w = gamma_w

  def correct_tip_resistance(self, qc, u2, a_net=0.80):
    """Eq. 5.16: qt = qc + (1 - a_net) * u2"""
    return qc + (1.0 - a_net) * u2

  def estimate_unit_weight(self, qt, fs, u2):
    """Averages three formulations from CFEM:

    Robertson & Cabal (2010), Mayne & Peuchen (2012), Mayne et al. (2022).
    """
    rf = np.clip((100.0 * fs) / np.maximum(qt, 1.0), 0.1, 15.0)

    gamma_rob = self.gamma_w * (
        0.27 * np.log10(rf)
        + 0.36 * np.log10(np.maximum(qt / self.p_atm, 0.01))
        + 1.236
    )

    gamma_mayne_fs = self.gamma_w * (
        1.22 + 0.345 * np.log10(np.maximum(100.0 * (fs / self.p_atm) + 0.01, 0.01))
    )

    qe = np.maximum(qt - u2, 0.1)
    gamma_mayne_qe = self.gamma_w * (1.54 + 0.254 * np.log10(qe / self.p_atm))

    gamma_avg = (gamma_rob + gamma_mayne_fs + gamma_mayne_qe) / 3.0
    return np.clip(gamma_avg, 12.0, 23.0)

  def calculate_stresses(self, depth, gamma_total, gwl=0.0):
    dz = np.diff(depth, prepend=depth[0])
    sigma_v0 = np.cumsum(gamma_total * dz)
    u0 = np.where(depth > gwl, (depth - gwl) * self.gamma_w, 0.0)
    sigma_v0_eff = np.maximum(sigma_v0 - u0, 1.0)
    return sigma_v0, u0, sigma_v0_eff

  def compute_normalized_parameters(
      self, qt, fs, u2, sigma_v0, u0, sigma_v0_eff
  ):
    q_net = qt - sigma_v0
    Q = np.maximum(q_net / sigma_v0_eff, 0.1)
    Fr = np.clip((fs / np.maximum(q_net, 1.0)) * 100.0, 0.01, 20.0)
    Bq = np.clip((u2 - u0) / np.maximum(q_net, 1.0), -0.5, 2.0)

    dim_term = np.maximum(Q * (1.0 - Bq) + 1.0, 0.1)
    term1 = 3.0 - np.log10(dim_term)
    term2 = 1.5 + 1.3 * np.log10(Fr)
    Ic = np.sqrt(term1**2 + term2**2)
    return Q, Fr, Bq, Ic

  def classify_sbtn(self, Q, Fr):
    zones = []
    zone_names = {
        1: "Sensitive fine grained",
        2: "Organic material",
        3: "Clay to silty clay",
        4: "Clayey silt to silty clay",
        5: "Silty sand to sandy silt",
        6: "Clean sand to silty sand",
        7: "Gravelly sand to sand",
        8: "Very stiff sand to clayey sand",
        9: "Very stiff fine grained",
    }
    for q_val, fr_val in zip(Q, Fr):
      if q_val < 10 and fr_val < 1.0:
        z = 1
      elif q_val < 10 and 1.0 <= fr_val < 1.8:
        z = 4
      elif q_val < 10 and fr_val >= 1.8:
        z = 3
      elif 10 <= q_val < 30 and fr_val >= 1.0:
        z = 4
      elif 10 <= q_val < 30 and fr_val < 1.0:
        z = 5
      elif 30 <= q_val < 100 and fr_val < 1.5:
        z = 6
      elif q_val >= 100 and fr_val < 0.8:
        z = 7
      elif q_val >= 30 and fr_val >= 1.5:
        z = 8
      elif q_val >= 10 and fr_val >= 3.0:
        z = 9
      else:
        z = 5
      zones.append(z)
    labels = [f"Zone {z}: {zone_names[z]}" for z in zones]
    return np.array(zones), labels

  def estimate_parameters(
      self, qt, sigma_v0, sigma_v0_eff, Bq, Ic, vs_arr=None
  ):
    q_net = np.maximum(qt - sigma_v0, 0.0)

    # Effective Friction Angle (Kulhawy & Mayne 1990)
    norm_ratio = (qt / self.p_atm) / np.sqrt(sigma_v0_eff / self.p_atm)
    phi_peak = 17.6 + 11.0 * np.log10(np.maximum(norm_ratio, 1.0))
    phi_peak = np.clip(phi_peak, 20.0, 48.0)

    # Undrained Shear Strength (Nkt from Bq)
    n_kt = 10.5 - 4.6 * np.log(np.maximum(Bq + 0.1, 0.01))
    n_kt = np.clip(n_kt, 6.0, 25.0)
    su = q_net / n_kt

    # OCR via yield stress (Eq. 5.41 & 5.43)
    m_prime = 1.0 - (0.28 / (1.0 + (Ic / 2.7) ** 30))
    sigma_p_eff = (
        0.33 * (q_net**m_prime) * ((self.p_atm / 100.0) ** (1.0 - m_prime))
    )
    ocr = np.clip(sigma_p_eff / sigma_v0_eff, 1.0, 30.0)

    out = {"phi_peak": phi_peak, "Nkt": n_kt, "su": su, "OCR": ocr}

    if vs_arr is not None:
      rho = 1900.0
      g0_kpa = (rho * (vs_arr**2)) / 1000.0
      out["G0_MPa"] = g0_kpa / 1000.0
    else:
      out["G0_MPa"] = np.nan

    return out


def generate_synthetic_data():
  depth = np.linspace(0.5, 20.0, 150)
  qc = np.where(
      depth <= 7.0,
      np.random.normal(9000, 600, len(depth)),
      np.random.normal(1500, 200, len(depth)),
  )
  fs = np.where(
      depth <= 7.0,
      qc * np.random.normal(0.007, 0.001, len(depth)),
      qc * np.random.normal(0.025, 0.002, len(depth)),
  )
  u2 = np.where(
      depth <= 7.0,
      depth * 9.81,
      depth * 9.81 + np.random.normal(160, 20, len(depth)),
  )
  vs = np.where(
      depth <= 7.0,
      np.random.normal(270, 10, len(depth)),
      np.random.normal(170, 10, len(depth)),
  )
  return pd.DataFrame(
      {"depth_m": depth, "qc_kPa": qc, "fs_kPa": fs, "u2_kPa": u2, "vs_m_s": vs}
  )


def build_cptu_plot(df, u0):
  fig = make_subplots(
      rows=1,
      cols=5,
      shared_yaxes=True,
      subplot_titles=(
          "Tip Resistance qt",
          "Sleeve Friction fs",
          "Pore Pressure u2",
          "Friction Angle / su",
          "SBTn Classification",
      ),
      horizontal_spacing=0.03,
  )

  # Column 1: qt
  fig.add_trace(
      go.Scatter(
          x=df["qt_kPa"],
          y=df["depth_m"],
          mode="lines",
          line=dict(color="#1f77b4", width=2),
          name="qt (kPa)",
      ),
      row=1,
      col=1,
  )

  # Column 2: fs
  fig.add_trace(
      go.Scatter(
          x=df["fs_kPa"],
          y=df["depth_m"],
          mode="lines",
          line=dict(color="#d62728", width=2),
          name="fs (kPa)",
      ),
      row=1,
      col=2,
  )

  # Column 3: u2 & u0
  fig.add_trace(
      go.Scatter(
          x=df["u2_kPa"],
          y=df["depth_m"],
          mode="lines",
          line=dict(color="#008080", width=2),
          name="u2 (kPa)",
      ),
      row=1,
      col=3,
  )
  fig.add_trace(
      go.Scatter(
          x=u0,
          y=df["depth_m"],
          mode="lines",
          line=dict(color="#333333", dash="dash", width=1.5),
          name="u0 (kPa)",
      ),
      row=1,
      col=3,
  )

  # Column 4: Strength (phi or su)
  fig.add_trace(
      go.Scatter(
          x=df["su_kPa"],
          y=df["depth_m"],
          mode="lines",
          line=dict(color="#ff7f0e", width=2),
          name="su (kPa)",
      ),
      row=1,
      col=4,
  )

  # Column 5: SBTn
  fig.add_trace(
      go.Scatter(
          x=df["SBTn_Zone"],
          y=df["depth_m"],
          mode="markers",
          marker=dict(
              size=5,
              color=df["SBTn_Zone"],
              colorscale="Viridis",
              showscale=False,
          ),
          text=df["SBTn_Description"],
          name="SBTn",
      ),
      row=1,
      col=5,
  )

  fig.update_yaxes(autorange="reversed", title_text="Depth (m)", row=1, col=1)
  fig.update_xaxes(title_text="kPa", row=1, col=1)
  fig.update_xaxes(title_text="kPa", row=1, col=2)
  fig.update_xaxes(title_text="kPa", row=1, col=3)
  fig.update_xaxes(title_text="kPa", row=1, col=4)
  fig.update_xaxes(
      title_text="Zone",
      tickmode="linear",
      tick0=1,
      dtick=1,
      range=[0.5, 9.5],
      row=1,
      col=5,
  )

  fig.update_layout(
      height=750,
      margin=dict(l=40, r=40, t=50, b=40),
      hovermode="y unified",
      showlegend=True,
  )
  return fig


def main():
  st.set_page_config(page_title="CFEM CPTu Dashboard", layout="wide")
  st.title("Automated CPTu Soil Characterization Platform")
  st.markdown("Compliant with Canadian Foundation Engineering Manual (CFEM)")

  # Sidebar Parameters
  st.sidebar.header("Investigation Parameters")
  a_net = st.sidebar.slider(
      "Net Area Ratio (a_net)",
      min_value=0.35,
      max_value=0.90,
      value=0.80,
      step=0.01,
  )
  gwl = st.sidebar.number_input(
      "Groundwater Level (m)", min_value=0.0, max_value=50.0, value=1.5, step=0.5
  )
  p_atm = st.sidebar.number_input(
      "Atmospheric Pressure (kPa)", value=101.325, step=1.0
  )

  st.sidebar.header("Input Data")
  uploaded_file = st.sidebar.file_uploader(
      "Upload CSV / Excel File", type=["csv", "xlsx"]
  )

  if uploaded_file is not None:
    try:
      if uploaded_file.name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file)
      else:
        df_raw = pd.read_excel(uploaded_file)
      st.sidebar.success("File uploaded successfully.")
    except Exception as e:
      st.sidebar.error(f"Error loading file: {e}")
      return
  else:
    st.sidebar.info("Using representative synthetic sounding data.")
    df_raw = generate_synthetic_data()

  # Check required columns
  req_cols = ["depth_m", "qc_kPa", "fs_kPa", "u2_kPa"]
  if not all(col in df_raw.columns for col in req_cols):
    st.error(f"Input file must contain the following columns: {req_cols}")
    return

  vs_data = df_raw["vs_m_s"].values if "vs_m_s" in df_raw.columns else None

  # Computation
  engine = CPTuEngine(p_atm=p_atm)
  depth = df_raw["depth_m"].values
  qc = df_raw["qc_kPa"].values
  fs = df_raw["fs_kPa"].values
  u2 = df_raw["u2_kPa"].values

  qt = engine.correct_tip_resistance(qc, u2, a_net=a_net)
  gamma_est = engine.estimate_unit_weight(qt, fs, u2)
  sigma_v0, u0, sigma_v0_eff = engine.calculate_stresses(
      depth, gamma_est, gwl=gwl
  )
  Q, Fr, Bq, Ic = engine.compute_normalized_parameters(
      qt, fs, u2, sigma_v0, u0, sigma_v0_eff
  )
  sbtn_zones, sbtn_labels = engine.classify_sbtn(Q, Fr)
  params = engine.estimate_parameters(
      qt, sigma_v0, sigma_v0_eff, Bq, Ic, vs_arr=vs_data
  )

  df_processed = pd.DataFrame({
      "depth_m": depth,
      "qc_kPa": qc,
      "fs_kPa": fs,
      "u2_kPa": u2,
      "qt_kPa": np.round(qt, 1),
      "gamma_kN_m3": np.round(gamma_est, 2),
      "sigma_v0_kPa": np.round(sigma_v0, 1),
      "sigma_v0_eff_kPa": np.round(sigma_v0_eff, 1),
      "u0_kPa": np.round(u0, 1),
      "Q": np.round(Q, 2),
      "Fr": np.round(Fr, 2),
      "Bq": np.round(Bq, 3),
      "Ic": np.round(Ic, 2),
      "SBTn_Zone": sbtn_zones,
      "SBTn_Description": sbtn_labels,
      "phi_peak_deg": np.round(params["phi_peak"], 1),
      "su_kPa": np.round(params["su"], 1),
      "OCR": np.round(params["OCR"], 1),
      "G0_MPa": np.round(params["G0_MPa"], 1),
  })

  # Visualization
  tab1, tab2 = st.tabs(["Sounding Profiles", "Tabular Dataset"])

  with tab1:
    fig = build_cptu_plot(df_processed, u0)
    st.plotly_chart(fig, use_container_width=True)

  with tab2:
    st.dataframe(df_processed, use_container_width=True)

  # Excel Export
  output = io.BytesIO()
  with pd.ExcelWriter(output, engine="openpyxl") as writer:
    df_processed.to_excel(writer, sheet_name="CPTu_Interpreted", index=False)
  excel_data = output.getvalue()

  st.sidebar.download_button(
      label="Download Excel Report",
      data=excel_data,
      file_name="CPTu_Interpreted_Report.xlsx",
      mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  )


if __name__ == "__main__":
  main()
