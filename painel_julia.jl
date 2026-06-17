using Plots
using Random
using Distributions

# 1. Configuração dos parâmetros físicos do Reator PWR
Random.seed!(42) # Garante a reprodutibilidade dos gráficos
num_simulacoes = 10000
lambda_medio = 1.45
limite_nucleo = 12.0

# 2. Geração dos dados para o Painel Inferior (Estatística de Cauda)
# K_eff nominal em torno de 0.998 com flutuações térmicas e de fabricação
d_keff = Normal(0.998, 0.003)
keff_cenarios = rand(d_keff, num_simulacoes)
limite_critico = 1.005
cenarios_criticos = keff_cenarios[keff_cenarios .>= limite_critico]
prob_cauda = length(cenarios_criticos) / num_simulacoes

# Cálculo das métricas para o texto final
media_keff = mean(keff_cenarios)
std_keff = std(keff_cenarios)
falhas = length(cenarios_criticos)

# 3. Configuração da Janela Gráfica (Os Dois Painéis)
# Criamos o layout de 2 linhas e 1 coluna
plot_layout = grid(2, 1)

# --- PAINEL SUPERIOR: Rastreamento Geométrico das Trajetórias ---
p1 = plot(title="Painel Superior: Amostra Visual do Transporte Estocástico (Caminhada Aleatória)",
          titlefont=font(10, :bold), xlabel="Posição X (cm)", ylabel="Posição Y (cm)",
          grid=true, gridstyle=:dot, gridalpha=0.6, aspect_ratio=:equal, legend=:topright)

cores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

for i in 1:5
    num_passos = rand(5:11)
    # Distribuição Exponencial em Julia usa o parâmetro de escala diretamente
    passos = rand(Exponential(lambda_medio), num_passos)
    
    # Gerando ângulos aleatórios
    angulos = rand(Uniform(0, 2π), num_passos)
    
    # Acumula as posições (cumsum)
    x_passos = cumsum(passos .* sin.(angulos))
    y_passos = cumsum(passos .* cos.(angulos))
    
    # Insere a origem (0,0) no início
    x = vcat(0.0, x_passos)
    y = vcat(0.0, y_passos)
    
    # Verifica o desfecho físico do último ponto
    dist_final = sqrt(x[end]^2 + y[end]^2)
    if dist_final > limite_nucleo
        label_fim = "Nêutron $i: Fuga"
        marcador = :xcross
    else
        label_fim = rand() > 0.4 ? "Nêutron $i: Fissão" : "Nêutron $i: Absorção"
        marcador = :circle
    end
    
    plot!(p1, x, y, linewidth=1.5, marker=marcador, color=cores[i], label=label_fim)
end

# Desenha a fronteira circular do núcleo
theta = range(0, 2π, length=100)
plot!(p1, limite_nucleo .* cos.(theta), limite_nucleo .* sin.(theta), 
      color=:black, linestyle=:dash, alpha=0.5, label="Fronteira do Núcleo")

# --- PAINEL INFERIOR: Distribuição Estatística de K_eff ---
p2 = histogram(keff_cenarios, bins=60, weights=ones(num_simulacoes)/num_simulacoes,
               color="#34495e", alpha=0.7, edgecolor=:white, label="Históricos MC",
               title="Painel Inferior: Análise Estatística de Cauda Larga (10.000 Cenários Coletados)",
               titlefont=font(10, :bold), xlabel="Fator de Multiplicação Efetivo (K_eff)",
               ylabel="Densidade de Probabilidade", grid=true, gridstyle=:dot, gridalpha=0.6, legend=:topright)

# Linha vertical indicando o limite de criticalidade imediata
vline!(p2, [limite_critico], color=:red, linewidth=2, label="Limite Supercrítico Prompt (K_eff = 1.005)")

# Monta a string de métricas exigida pela professora
texto_metricas = "Média K_eff: $(round(media_keff, digits=5))\n" *
                 "Desvio Padrão: $(round(std_keff, digits=5))\n" *
                 "Cenários de Falha: $falhas\n" *
                 "Prob. Cauda: $(round(prob_cauda, digits=6))\n" *
                 "Veredicto: REJEITADO (Risco > 10^-6)"

# Adiciona o texto anotado no gráfico inferior
# Ajuste as coordenadas x e y conforme os limites gerados pelos dados
annotate!(p2, 0.990, 0.05, text(texto_metricas, 8, :left, :black))

# --- JUNTANDO OS DOIS PAINÉIS ---
meu_painel = plot(p1, p2, layout=plot_layout, size=(800, 800), 
                  plot_title="Simulador de Caminhada Aleatória de Nêutrons e Painel Analítico", plot_titlefont=font(13, :bold))

# CORREÇÃO: Salva o painel automaticamente na mesma pasta do script
savefig(meu_painel, "painel_reator.png")
println("Gráfico gerado com sucesso e salvo como 'painel_reator.png'!")
