import numpy as np
import matplotlib.pyplot as plt

# 1. Configuração dos parâmetros físicos do Reator PWR
np.random.seed(42) # Garante a reprodutibilidade dos gráficos
num_simulacoes = 10000
lambda_medio = 1.45
limite_nucleo = 12.0

# 2. Geração dos dados para o Painel Inferior (Estatística de Cauda)
# K_eff nominal em torno de 0.998 com flutuações térmicas e de fabricação
keff_cenarios = np.random.normal(loc=0.998, scale=0.003, size=num_simulacoes)
limite_critico = 1.005
cenarios_criticos = keff_cenarios[keff_cenarios >= limite_critico]
prob_cauda = len(cenarios_criticos) / num_simulacoes

# 3. Configuração da Janela Gráfica (Os Dois Painéis)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle("Simulador de Caminhada Aleatória de Nêutrons e Painel Analítico", fontsize=14, fontweight='bold')

# --- PAINEL SUPERIOR: Rastreamento Geométrico das Trajetórias ---
# Simula a caminhada aleatória (Random Walk) de 5 nêutrons de exemplo para a aula
cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i in range(5):
    # Cada nêutron dá passos baseados na distribuição exponencial do livre caminho médio
    num_passos = np.random.randint(5, 12)
    passos = np.random.exponential(scale=lambda_medio, size=num_passos)
    
    # Acumula as posições para criar a trajetória tridimensional projetada em 2D
    x = np.insert(np.cumsum(passos * np.sin(np.random.uniform(0, 2*np.pi, num_passos))), 0, 0)
    y = np.insert(np.cumsum(passos * np.cos(np.random.uniform(0, 2*np.pi, num_passos))), 0, 0)
    
    # Verifica o desfecho físico do último ponto
    dist_final = np.sqrt(x[-1]**2 + y[-1]**2)
    if dist_final > limite_nucleo:
        label_fim = f"Nêutron {i+1}: Fuga"
        marcador = 'x'
    else:
        label_fim = f"Nêutron {i+1}: Fissão" if np.random.rand() > 0.4 else f"Nêutron {i+1}: Absorção"
        marcador = 'o'
        
    # CORREÇÃO: Alterado 'borderwidth' para 'linewidth'
    ax1.plot(x, y, linewidth=1.5, marker=marcador, color=cores[i], label=label_fim)

# Desenha o limite físico do núcleo do reator PWR
circulo_nucleo = plt.Circle((0, 0), limite_nucleo, color='black', fill=False, linestyle='--', alpha=0.5, label="Fronteira do Núcleo")
ax1.add_patch(circulo_nucleo)
ax1.set_title("Painel Superior: Amostra Visual do Transporte Estocástico (Caminhada Aleatória)", fontsize=11, fontweight='bold')
ax1.set_xlabel("Posição X (cm)")
ax1.set_ylabel("Posição Y (cm)")
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc="upper right", fontsize=8)
ax1.axis('equal')

# --- PAINEL INFERIOR: Distribuição Estatística de K_eff ---
# Plota o histograma de densidade probabilística dos 10.000 cenários
n, bins, patches = ax2.hist(keff_cenarios, bins=60, density=True, color='#34495e', alpha=0.7, edgecolor='white', label="Históricos MC")

# Destaca a região crítica de cauda (K_eff >= 1.005) em vermelho
for patch, left_bin in zip(patches, bins[:-1]):
    if left_bin >= limite_critico:
        patch.set_facecolor('#e74c3c')
        patch.set_alpha(0.9)

# CORREÇÃO: Alterado '#red' para 'red'
ax2.axvline(limite_critico, color='red', linestyle='-', linewidth=2, label="Limite Supercrítico Prompt (K_eff = 1.005)")
ax2.set_title("Painel Inferior: Análise Estatística de Cauda Larga (10.000 Cenários Coletados)", fontsize=11, fontweight='bold')
ax2.set_xlabel("Fator de Multiplicação Efetivo (K_eff)")
ax2.set_ylabel("Densidade de Probabilidade")
ax2.grid(True, linestyle=':', alpha=0.6)

# Caixa de texto com as métricas que a professora exigiu para o veredicto regulatório
texto_metricas = (f"Média K_eff: {np.mean(keff_cenarios):.5f}\n"
                  f"Desvio Padrão: {np.std(keff_cenarios):.5f}\n"
                  f"Cenários de Falha: {len(cenarios_criticos)}\n"
                  f"Prob. Cauda: {prob_cauda:.6f}\n"
                  f"Veredicto: REJEITADO (Risco > 10^-6)")
ax2.text(0.02, 0.95, texto_metricas, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
ax2.legend(loc="upper right", fontsize=8)

# Ajusta o espaçamento e exibe os painéis combinados
plt.tight_layout()
plt.show()
