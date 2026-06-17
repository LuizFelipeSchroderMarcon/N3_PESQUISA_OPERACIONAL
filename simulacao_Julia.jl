using Base.Threads
using Random
using Statistics

# Definição das estruturas imutáveis para garantir desempenho e concorrência segura
struct ParametrosNucleo
    λ_medio::Float64          # Livre caminho médio padrão
    Σ_abs_media::Float64      # Seção de choque macroscópica de absorção média
    σ_temperatura::Float64    # Desvio padrão induzido pela oscilação térmica
    enriquecimento_medio::Float64 # Fração nominal de U-235
    σ_fabricacao::Float64     # Margem de erro de fabricação do combustível
    prob_nu::Vector{Float64}   # Distribuição discreta de probabilidade para ν (0, 1, 2, 3, 4)
end

"""
    simular_neutron(param::ParametrosNucleo)

Simula o ciclo de vida completo de uma única partícula de nêutron dentro do núcleo 
e retorna o saldo local de geração versus perda para compor o K_eff daquele cenário.
"""
function simular_neutron(param::ParametrosNucleo)::Float64
    # 1. Amostragem das variáveis de fabricação e ambiente locais para o histórico
    enriquecimento_local = param.enriquecimento_medio + randn() * param.σ_fabricacao
    Σ_abs_local = param.Σ_abs_media + randn() * param.σ_temperatura
    
    # 2. Amostragem do Livre Caminho Médio de Colisão (Distribuição Exponencial)
    # d = -λ * ln(R)
    distancia_voo = -param.λ_medio * log(rand())
    
    # 3. Critério de Fuga Geométrica Simplificado 
    # Se a distância ultrapassar as fronteiras físicas simuladas do reator, o nêutron escapa
    if distancia_voo > 12.0
        return 0.0 # Nêutron perdido por fuga geométrica, não gera fissão
    end
    
    # 4. Determinação do desfecho da colisão com base nas seções de choque e enriquecimento
    # Fator probabilístico de interação competitiva
    fator_interacao = rand() * (Σ_abs_local / 0.5)
    
    if fator_interacao * (1.0 / enriquecimento_local) > 0.65
        # O nêutron foi capturado e absorvido pelas barras de controle ou moderador
        return 0.0
    else
        # Evento de Fissão Nuclear validado: Sorteia o número discreto de nêutrons secundários (ν)
        r = rand()
        acumulado = 0.0
        nu_sorteado = 0
        
        for i in 1:length(param.prob_nu)
            acumulado += param.prob_nu[i]
            if r <= acumulado
                nu_sorteado = i - 1 # Índices em Julia começam em 1, ν varia de 0 a 4
                break
            end
        end
        return Float64(nu_sorteado)
    end
end

"""
    executar_monte_carlo(num_cenarios::Int, param::ParametrosNucleo)

Gerencia o loop global de Monte Carlo distribuindo o processamento dos históricos
de forma concorrente através das threads disponíveis no runtime de Julia.
"""
function executar_monte_carlo(num_cenarios::Int, param::ParametrosNucleo)
    # Vetor compartilhado pré-alocado para armazenar o resultado de cada cenário
    resultados_keff = Vector{Float64}(undef, num_cenarios)
    
    # Loop concorrente paralelo multi-thread nativo
    Threads.@threads for i in 1:num_cenarios
        # Cada histórico rastreia a resposta local do ciclo do nêutron
        # O K_eff local é a razão de novos nêutrons gerados por nêutron inicial (1.0)
        neutrons_gerados = simular_neutron(param)
        resultados_keff[i] = neutrons_gerados / 1.0 
    end
    
    return resultados_keff
end

"""
    exibir_painel_analitico(resultados::Vector{Float64})

Processa a redução estatística dos dados calculando parâmetros de cauda e 
exibe os relatórios de validação regulatória exigidos pela pós-graduação.
"""
function exibir_painel_analitico(resultados::Vector{Float64})
    num_simulacoes = length(resultados)
    keff_medio = mean(resultados)
    desvio_padrao = std(resultados)
    
    # Cálculo preciso da probabilidade de cauda (K_eff >= 1.005)
    cenarios_supercriticos = count(k -> k >= 1.005, resultados)
    probabilidade_cauda = cenarios_supercriticos / num_simulacoes
    
    # Limite regulatório estipulado: 1 em 1.000.000 (1e-6)
    limite_regulatorio = 1 / 1000000
    status_seguranca = probabilidade_cauda <= limite_regulatorio ? "APROVADO (Geometria Robusta)" : "REJEITADO (Risco de Excursão)"
    
    println("======================================================================")
    println("      PAINEL ANALÍTICO CONCORRENTE - SIMULAÇÃO DE MONTE CARLO         ")
    println("======================================================================")
    println("Total de Históricos de Nêutrons Processados : ", num_simulacoes)
    println("Fator de Multiplicação Efetivo Médio (K_eff): ", round(keff_medio, digits=5))
    println("Desvio Padrão Estatístico de K_eff          : ", round(desvio_padrao, digits=5))
    println("Cenários Supercríticos Prompt Detectados    : ", cenarios_supercriticos)
    println("Probabilidade de Cauda Calculada            : ", probabilidade_cauda)
    println("Limite Regulatório de Segurança Aceitável   : ", limite_regulatorio)
    println("----------------------------------------------------------------------")
    println("VEREDICTO DO DESIGN DO NÚCLEO               : ", status_seguranca)
    println("======================================================================")
end

# --- Inicialização da Execução ---
function main()
    # Definição dos dados e perfis físicos fornecidos para o Reator PWR
    # Probabilidades discretas para ν = [0, 1, 2, 3, 4] respectivamente
    distribuicao_nu = [0.05, 0.15, 0.25, 0.45, 0.10] 
    
    config_pwr = ParametrosNucleo(
        1.45,           # λ_medio (Livre caminho médio nominal)
        0.22,           # Σ_abs_media (Seção de choque macroscópica de absorção nominal)
        0.015,          # σ_temperatura (Oscilação térmica)
        0.045,          # enriquecimento_medio (4.5% de Urânio 235)
        0.001,          # σ_fabricacao (Margem de erro dimensional)
        distribuicao_nu # Distribuição discreta de nêutrons por fissão
    )
    
    # Escopo solicitado: Execução massiva com 100.000 cenários de amostragem
    cenarios_solicitados = 100000
    
    # Execução do núcleo de processamento estocástico
    historicos = executar_monte_carlo(cenarios_solicitados, config_pwr)
    
    # Apresentação dos resultados consolidados
    exibir_painel_analitico(historicos)
end

# Executa o programa
main()