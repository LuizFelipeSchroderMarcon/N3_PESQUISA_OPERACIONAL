import numpy as np

def rodar_simulacao(n_simulacoes):

    rng = np.random.default_rng(42)

    resultados = []

    for _ in range(n_simulacoes):

        lambda_ = rng.normal(1.00, 0.03)
        sigma = rng.normal(1.00, 0.02)
        u235 = rng.normal(1.00, 0.01)
        nu = rng.normal(1.00, 0.015)

        k_eff = (nu * u235) / (lambda_ * sigma)

        resultados.append(k_eff)

    return np.array(resultados)


# EXECUÇÃO

resultados = rodar_simulacao(20000)

k_medio = np.mean(resultados)

prob_supercritico = (
    np.sum(resultados >= 1.005)
    / len(resultados)
) * 100

print("=== RESULTADOS DA SIMULAÇÃO ===")
print(f"Número de simulações: 20000")
print(f"k_eff médio: {k_medio:.6f}")
print(f"Probabilidade supercrítica: {prob_supercritico:.2f}%")
print(f"Maior k_eff: {np.max(resultados):.6f}")
print(f"Menor k_eff: {np.min(resultados):.6f}")
print(f"Desvio padrão: {np.std(resultados):.6f}")