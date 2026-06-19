# Comando para instalar bibliotecas: python -m pip install streamlit numpy pandas plotly
# Comando para rodar o código: python -m streamlit run simulador_ev.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import heapq

# ─────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Simulação Monte Carlo — Postos de Recarga EV",
    page_icon="⚡",
    layout="wide",
)

# ─────────────────────────────────────────────
# CSS customizado
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background-color: #0e1117; }

    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #252b3b);
        border: 1px solid #2d3a4f;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00d4ff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8892a4;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #4caf50;
        margin-top: 6px;
        font-weight: 500;
    }

    .recommendation-box {
        background: linear-gradient(135deg, #0d2137, #0a2f1f);
        border: 2px solid #00d4ff;
        border-radius: 14px;
        padding: 22px 28px;
        margin: 16px 0;
    }
    .rec-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #00d4ff;
        margin-bottom: 8px;
    }
    .rec-text {
        font-size: 0.95rem;
        color: #cdd6e0;
        line-height: 1.6;
    }

    h1, h2, h3 { color: #e8edf2 !important; }
    .stSlider label { color: #8892a4 !important; }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #00d4ff;
        border-left: 4px solid #00d4ff;
        padding-left: 12px;
        margin: 24px 0 14px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Funções de simulação
# ─────────────────────────────────────────────
TAXA_CHEGADA_POR_HORA = {
    0: 1, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 1,
    6: 4,  7: 7,   8: 8,   9: 5,   10: 4,  11: 4,
    12: 5, 13: 4,  14: 4,  15: 5,  16: 7,  17: 10,
    18: 9, 19: 8,  20: 6,  21: 4,  22: 3,  23: 2
}

TARIFA_POR_HORA = {
    h: 0.40 + 0.35 * np.sin(np.pi * (h - 6) / 14) + np.random.normal(0, 0.03)
    for h in range(24)
}

def gerar_chegadas_dia(rng):
    """Gera todos os carros de um dia de forma vetorizada (rápido)."""
    horas = np.array(list(TAXA_CHEGADA_POR_HORA.keys()))
    taxas = np.array(list(TAXA_CHEGADA_POR_HORA.values()))

    n_por_hora = rng.poisson(taxas)
    total = n_por_hora.sum()
    if total == 0:
        return np.array([]), np.array([]), np.array([])

    horas_expandidas = np.repeat(horas, n_por_hora)
    chegada = horas_expandidas * 60 + rng.uniform(0, 60, size=total)

    # Nível de bateria na chegada (carros raramente chegam totalmente vazios)
    soc_inicial = rng.beta(2, 5, size=total) * 70 + 10        # 10% a 80%
    # Carregador rápido tipicamente carrega até ~80-90% (depois a taxa cai muito)
    soc_alvo = rng.uniform(70, 90, size=total)
    cap_bateria = rng.choice([40, 60, 77, 100], size=total)   # kWh
    potencia_carregador = 100                                 # kW (carregador DC rápido)

    delta_soc = np.maximum(soc_alvo - soc_inicial, 5)         # mínimo 5% de carga
    tempo_servico = delta_soc / 100 * cap_bateria / potencia_carregador * 60
    tempo_servico = np.clip(tempo_servico, 5, 60)             # entre 5 e 60 min

    ordem = np.argsort(chegada)
    return chegada[ordem], tempo_servico[ordem], horas_expandidas[ordem]


def simular_dia(n_totens, rng):
    """Simula um dia completo no posto de recarga (com heap para os totens)."""
    chegada, tempo_servico, horas = gerar_chegadas_dia(rng)
    n = len(chegada)
    if n == 0:
        return np.array([]), np.zeros(24)

    livres_em = [0.0] * n_totens
    heapq.heapify(livres_em)

    tempos_espera = np.empty(n)
    ocupacao_hora = np.zeros(24)

    for i in range(n):
        proximo_livre = heapq.heappop(livres_em)
        t_inicio = max(chegada[i], proximo_livre)
        tempos_espera[i] = t_inicio - chegada[i]
        heapq.heappush(livres_em, t_inicio + tempo_servico[i])
        ocupacao_hora[int(horas[i])] += tempo_servico[i] / 60

    return tempos_espera, ocupacao_hora


@st.cache_data(show_spinner=False)
def rodar_monte_carlo(n_totens, n_iteracoes, seed=42):
    """Roda n_iteracoes simulações e agrega os resultados. Resultado fica em cache."""
    rng = np.random.default_rng(seed)

    todos_tempos = []
    ocupacao_media = np.zeros(24)

    for _ in range(n_iteracoes):
        tempos, ocupacao = simular_dia(n_totens, rng)
        todos_tempos.append(tempos)
        ocupacao_media += ocupacao

    todos_tempos = np.concatenate(todos_tempos)
    ocupacao_media /= n_iteracoes
    return todos_tempos, ocupacao_media


@st.cache_data(show_spinner=False)
def calcular_prob_por_totens(max_totens, n_iteracoes, limite_min=10, seed=42):
    """Calcula P(espera < limite) para cada número de totens. Resultado fica em cache."""
    resultados = []
    for c in range(1, max_totens + 1):
        tempos, _ = rodar_monte_carlo(c, n_iteracoes, seed=seed)
        prob = np.mean(tempos < limite_min) * 100
        media = np.mean(tempos)
        p95 = np.percentile(tempos, 95)
        resultados.append({
            "totens": c,
            "prob_ok": prob,
            "media_espera": media,
            "p95_espera": p95
        })
    return pd.DataFrame(resultados)


# ─────────────────────────────────────────────
# Interface — Header
# ─────────────────────────────────────────────
col_logo, col_titulo = st.columns([1, 8])
with col_logo:
    st.markdown("<div style='font-size:3.5rem; padding-top:10px'>⚡</div>", unsafe_allow_html=True)
with col_titulo:
    st.markdown("## Simulação Monte Carlo — Postos de Recarga EV")
    st.markdown(
        "<span style='color:#8892a4; font-size:0.95rem'>"
        "Pesquisa Operacional · Modelo de Fila M/G/c com variáveis estocásticas"
        "</span>",
        unsafe_allow_html=True
    )

st.divider()

# ─────────────────────────────────────────────
# Sidebar — Parâmetros
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parâmetros da Simulação")
    st.markdown("---")

    with st.form("parametros_form"):
        n_totens = st.slider("Número de totens de recarga", 1, 8, 3,
                             help="Quantidade de carregadores instalados no posto")

        n_iteracoes = st.select_slider(
            "Iterações Monte Carlo",
            options=[100, 500, 1000, 2000, 5000],
            value=500,
            help="Mais iterações = maior precisão, mas mais lento"
        )

        limite_espera = st.slider("Limite aceitável de espera (min)", 5, 30, 10,
                                  help="Critério de nível de serviço")

        st.markdown("---")
        st.markdown("### 📊 Análise Comparativa")
        max_totens_comp = st.slider("Comparar até N totens", 2, 8, 5)

        rodar = st.form_submit_button("▶ Rodar Simulação", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# Execução da simulação principal
# ─────────────────────────────────────────────
if rodar or "resultados" not in st.session_state:
    with st.spinner("Rodando simulação Monte Carlo..."):
        tempos_espera, ocupacao_hora = rodar_monte_carlo(n_totens, n_iteracoes)
        df_comp = calcular_prob_por_totens(max_totens_comp, n_iteracoes, limite_espera)

    st.session_state["resultados"] = True
    st.session_state["tempos_espera"] = tempos_espera
    st.session_state["ocupacao_hora"] = ocupacao_hora
    st.session_state["df_comp"] = df_comp
    st.session_state["params"] = (n_totens, n_iteracoes, limite_espera)
else:
    tempos_espera = st.session_state["tempos_espera"]
    ocupacao_hora = st.session_state["ocupacao_hora"]
    df_comp = st.session_state["df_comp"]
    n_totens, n_iteracoes, limite_espera = st.session_state["params"]

# ─────────────────────────────────────────────
# Métricas principais
# ─────────────────────────────────────────────
prob_ok = np.mean(tempos_espera < limite_espera) * 100
media_espera = np.mean(tempos_espera)
p95_espera = np.percentile(tempos_espera, 95)
total_carros = len(tempos_espera)

st.markdown("<div class='section-title'>📈 Resultados para {} Totem(ns)</div>".format(n_totens), unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
cards = [
    (f"{prob_ok:.1f}%", f"Atendidos em < {limite_espera} min", "Meta: ≥ 90%"),
    (f"{media_espera:.1f} min", "Tempo médio de espera", "Média das simulações"),
    (f"{p95_espera:.1f} min", "Espera no percentil 95", "Pior cenário frequente"),
    (f"{total_carros // n_iteracoes}", "Veículos/dia (média)", f"{n_iteracoes} dias simulados"),
]
for col, (val, label, sub) in zip([c1, c2, c3, c4], cards):
    with col:
        cor_val = "#00d4ff" if label != f"Atendidos em < {limite_espera} min" else (
            "#4caf50" if prob_ok >= 90 else "#ff5252"
        )
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value' style='color:{cor_val}'>{val}</div>
            <div class='metric-label'>{label}</div>
            <div class='metric-sub'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

# Recomendação
totem_ideal_row = df_comp[df_comp["prob_ok"] >= 90]
if len(totem_ideal_row) > 0:
    totem_ideal = int(totem_ideal_row["totens"].min())
    status_icon = "✅" if n_totens >= totem_ideal else "⚠️"
    msg = (f"O número <b>ideal de totens é {totem_ideal}</b>, garantindo que "
           f"{df_comp[df_comp['totens']==totem_ideal]['prob_ok'].values[0]:.1f}% dos veículos esperem menos de {limite_espera} minutos. "
           f"{'Configuração atual está adequada! ✅' if n_totens >= totem_ideal else f'Configuração atual ({n_totens} totens) está abaixo do ideal.'}")
else:
    msg = f"Nenhuma configuração de até {max_totens_comp} totens atingiu 90% com o critério de {limite_espera} min. Considere aumentar o limite ou o número máximo de totens."

st.markdown(f"""
<div class='recommendation-box'>
    <div class='rec-title'>🎯 Recomendação da Simulação</div>
    <div class='rec-text'>{msg}</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Gráficos — Linha 1
# ─────────────────────────────────────────────
st.markdown("<div class='section-title'>📊 Análise Detalhada</div>", unsafe_allow_html=True)

col_g1, col_g2 = st.columns(2)

# Gráfico 1 — Histograma de espera
with col_g1:
    st.markdown("**Distribuição do Tempo de Espera**")
    bins = np.linspace(0, min(tempos_espera.max(), 60), 40)
    hist, edges = np.histogram(tempos_espera, bins=bins)
    cores = ["#ff5252" if (edges[i] + edges[i+1])/2 > limite_espera else "#00d4ff"
             for i in range(len(hist))]

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=[(edges[i] + edges[i+1])/2 for i in range(len(hist))],
        y=hist / hist.sum() * 100,
        marker_color=cores,
        name="Frequência",
        hovertemplate="Espera: %{x:.1f} min<br>Frequência: %{y:.1f}%<extra></extra>"
    ))
    fig1.add_vline(x=limite_espera, line_dash="dash", line_color="#ffd700",
                   annotation_text=f"Limite {limite_espera} min", annotation_font_color="#ffd700")
    fig1.update_layout(
        plot_bgcolor="#1a1f2e", paper_bgcolor="#1a1f2e",
        font_color="#cdd6e0", height=320,
        xaxis_title="Tempo de espera (min)",
        yaxis_title="Frequência (%)",
        showlegend=False,
        margin=dict(l=40, r=20, t=20, b=40)
    )
    fig1.update_xaxes(gridcolor="#2d3a4f")
    fig1.update_yaxes(gridcolor="#2d3a4f")
    st.plotly_chart(fig1, use_container_width=True)

# Gráfico 2 — Prob vs Totens
with col_g2:
    st.markdown("**Probabilidade de Atendimento por Nº de Totens**")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df_comp["totens"],
        y=df_comp["prob_ok"],
        mode="lines+markers",
        line=dict(color="#00d4ff", width=3),
        marker=dict(size=10, color=[
            "#4caf50" if v >= 90 else "#ff5252" for v in df_comp["prob_ok"]
        ], line=dict(color="#0e1117", width=2)),
        name="P(espera < limite)",
        hovertemplate="Totens: %{x}<br>Prob.: %{y:.1f}%<extra></extra>"
    ))
    fig2.add_hline(y=90, line_dash="dash", line_color="#ffd700",
                   annotation_text="Meta 90%", annotation_font_color="#ffd700")
    fig2.add_vline(x=n_totens, line_dash="dot", line_color="#9c88ff",
                   annotation_text=f"Atual ({n_totens})", annotation_font_color="#9c88ff")
    fig2.update_layout(
        plot_bgcolor="#1a1f2e", paper_bgcolor="#1a1f2e",
        font_color="#cdd6e0", height=320,
        xaxis_title="Número de totens",
        yaxis_title=f"P(espera < {limite_espera} min) %",
        showlegend=False,
        xaxis=dict(tickmode="linear", dtick=1, gridcolor="#2d3a4f"),
        yaxis=dict(range=[0, 105], gridcolor="#2d3a4f"),
        margin=dict(l=40, r=20, t=20, b=40)
    )
    st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────
# Gráficos — Linha 2
# ─────────────────────────────────────────────
col_g3, col_g4 = st.columns(2)

# Gráfico 3 — Ocupação por hora
with col_g3:
    st.markdown("**Ocupação dos Carregadores por Hora do Dia**")
    taxa_chegada = list(TAXA_CHEGADA_POR_HORA.values())
    ocupacao_pct = np.minimum(ocupacao_hora / n_totens, 1.0) * 100  # taxa real por totem, em %

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=list(range(24)),
        y=taxa_chegada,
        name="Chegadas/hora (λ)",
        marker_color="#9c88ff",
        opacity=0.7,
        yaxis="y2",
        hovertemplate="Hora: %{x}h<br>λ: %{y} carros/h<extra></extra>"
    ))
    fig3.add_trace(go.Scatter(
        x=list(range(24)),
        y=ocupacao_pct,
        mode="lines+markers",
        name="Ocupação por totem (%)",
        line=dict(color="#00d4ff", width=2.5),
        marker=dict(size=6),
        hovertemplate="Hora: %{x}h<br>Ocupação: %{y:.1f}%<extra></extra>"
    ))
    fig3.update_layout(
        plot_bgcolor="#1a1f2e", paper_bgcolor="#1a1f2e",
        font_color="#cdd6e0", height=320,
        xaxis=dict(tickmode="linear", dtick=2, title="Hora do dia", gridcolor="#2d3a4f"),
        yaxis=dict(title="Ocupação média por totem (%)", range=[0, 105], gridcolor="#2d3a4f"),
        yaxis2=dict(title="Chegadas/hora (λ)", overlaying="y", side="right",
                    showgrid=False, tickfont=dict(color="#9c88ff")),
        legend=dict(orientation="h", y=-0.25, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=60, t=20, b=60)
    )
    st.plotly_chart(fig3, use_container_width=True)

# Gráfico 4 — Convergência
with col_g4:
    st.markdown("**Convergência da Simulação**")
    np.random.seed(42)
    amostras = np.random.choice(tempos_espera, size=min(len(tempos_espera), 2000), replace=False)
    medias_acum = np.cumsum(amostras) / np.arange(1, len(amostras) + 1)
    intervalo = 1.96 * np.std(amostras) / np.sqrt(np.arange(1, len(amostras) + 1))

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=list(range(1, len(medias_acum) + 1)),
        y=medias_acum + intervalo,
        fill=None, mode="lines",
        line=dict(width=0), showlegend=False
    ))
    fig4.add_trace(go.Scatter(
        x=list(range(1, len(medias_acum) + 1)),
        y=np.maximum(medias_acum - intervalo, 0),
        fill="tonexty", mode="lines",
        line=dict(width=0),
        fillcolor="rgba(0, 212, 255, 0.15)",
        name="IC 95%"
    ))
    fig4.add_trace(go.Scatter(
        x=list(range(1, len(medias_acum) + 1)),
        y=medias_acum,
        mode="lines",
        line=dict(color="#00d4ff", width=2),
        name="Média acumulada"
    ))
    fig4.add_hline(y=media_espera, line_dash="dash", line_color="#4caf50",
                   annotation_text=f"Convergência: {media_espera:.1f} min",
                   annotation_font_color="#4caf50")
    fig4.update_layout(
        plot_bgcolor="#1a1f2e", paper_bgcolor="#1a1f2e",
        font_color="#cdd6e0", height=320,
        xaxis_title="Número de iterações",
        yaxis_title="Espera média (min)",
        xaxis=dict(gridcolor="#2d3a4f"),
        yaxis=dict(gridcolor="#2d3a4f"),
        legend=dict(orientation="h", y=-0.25, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=20, b=60)
    )
    st.plotly_chart(fig4, use_container_width=True)

# ─────────────────────────────────────────────
# Tabela comparativa
# ─────────────────────────────────────────────
st.markdown("<div class='section-title'>📋 Tabela Comparativa por Número de Totens</div>", unsafe_allow_html=True)

df_display = df_comp.copy()
df_display.columns = ["Totens", f"P(espera < {limite_espera} min) %", "Espera Média (min)", "Percentil 95 (min)"]
df_display[f"P(espera < {limite_espera} min) %"] = df_display[f"P(espera < {limite_espera} min) %"].round(1)
df_display["Espera Média (min)"] = df_display["Espera Média (min)"].round(2)
df_display["Percentil 95 (min)"] = df_display["Percentil 95 (min)"].round(2)
df_display["Atinge Meta?"] = df_display[f"P(espera < {limite_espera} min) %"].apply(
    lambda x: "✅ Sim" if x >= 90 else "❌ Não"
)

st.dataframe(
    df_display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Totens": st.column_config.NumberColumn(format="%d"),
        f"P(espera < {limite_espera} min) %": st.column_config.ProgressColumn(
            min_value=0, max_value=100, format="%.1f%%"
        ),
    }
)

# ─────────────────────────────────────────────
# Rodapé
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#4a5568; font-size:0.8rem'>"
    "Simulação Monte Carlo · Modelo de Fila M/G/c · Pesquisa Operacional"
    "</div>",
    unsafe_allow_html=True
)