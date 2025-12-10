import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from typing import List, Dict, Tuple
import os

st.set_page_config(page_title="PDM Projetos Sociais", layout="wide")

# =============================
# UTILIDADES COMUNS
# =============================
def prepare_evaluation(decision_makers: List[str], analysts: List[str], experts: List[str]) -> Dict:
    return {"decision_makers": decision_makers, "analysts": analysts, "experts": experts}

def optimize_criteria_default(criteria: List[str]) -> Dict:
    opt = {}
    for c in criteria:
        if c in predefined_criteria:
            tipo = predefined_criteria[c]['type']
            opt[c] = "Minimize" if "(min)" in tipo else "Maximize"
        else:
            if any(key in c.lower() for key in ["impact", "benefício", "eficácia", "resultado", "aceitação", "satisfação", "emprego", "retorno"]):
                opt[c] = "Maximize"
            else:
                opt[c] = "Minimize"
    return opt

def build_consequence_matrix(projects: List[str], criteria: List[str], seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=50, scale=10, size=(len(projects), len(criteria)))
    df = pd.DataFrame(data=np.maximum(0, data), index=projects, columns=criteria).round(2)
    return df

def compute_preference_flows(matrix: np.ndarray, criteria: List[str], optimization: Dict,
                             weights: List[float], preference_functions: Dict, thresholds: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_projects, _ = matrix.shape
    phi_plus = np.zeros(n_projects)
    phi_minus = np.zeros(n_projects)

    # Normaliza os pesos para garantir que a soma seja 1, especialmente útil para sensibilidade
    sum_w = sum(weights)
    if sum_w == 0:
        # Se os pesos somam 0, retorna fluxos zero.
        return np.zeros(n_projects), np.zeros(n_projects), np.zeros(n_projects), np.zeros(n_projects)
    
    normalized_weights = [w / sum_w for w in weights]
    
    for i in range(n_projects):
        for j in range(n_projects):
            if i == j:
                continue
            sum_pref = 0.0
            for k, c in enumerate(criteria):
                # Diferença (d = g(a) - g(b) se max, g(b) - g(a) se min)
                d = matrix[i, k] - matrix[j, k] if optimization[c] == "Maximize" else matrix[j, k] - matrix[i, k]
                func = preference_functions[c]
                q = thresholds.get(c, {}).get('q', 0)
                p = thresholds.get(c, {}).get('p', 1.0)
                
                pref = 0.0
                if d > 0:
                    if func == 'u': # Usual
                        pref = 1.0
                    elif func == 'v': # V-Shape
                        pref = min(1.0, d / p)
                    elif func == 'l': # Linear
                        pref = 0.0 if d <= q else min(1.0, (d - q) / (p - q))
                
                sum_pref += normalized_weights[k] * pref
            
            # Promethee I e II usam a soma total das preferências/número de critérios, mas 
            # na verdade, o fluxo é a média sobre o número de projetos (n-1). 
            # Já que os pesos são normalizados, a preferência global é simplesmente sum_pref.
            # Contudo, mantendo a estrutura original para o fluxo:
            pref_overall = sum_pref 
            
            phi_plus[i] += pref_overall
            phi_minus[j] += pref_overall
            
    # Na implementação padrão, os fluxos são somados sobre todos os (n-1) projetos.
    # Como a soma_pref já incorpora a ponderação dos critérios (peso), 
    # a divisão por (n_projects - 1) não se aplica quando w_i é usado em vez de w_i/(n_projects-1). 
    # Aqui, a soma sobre os projetos já é o cálculo do Promethee I.
    
    phi_net = phi_plus - phi_minus
    T = max(0.0, -np.min(phi_net)) if len(phi_net) > 0 else 0.0
    phi_star = phi_net + T
    
    return phi_plus, phi_minus, phi_net, phi_star

def promethee_v_c_otimo(matrix: np.ndarray, projects: List[str], criteria: List[str],
                        optimization: Dict, weights: List[float], preference_functions: Dict,
                        thresholds: Dict, budget_constraint: float, cost_criterion_idx: int):
    
    phi_plus, phi_minus, phi_net, phi_star = compute_preference_flows(
        matrix, criteria, optimization, weights, preference_functions, thresholds
    )
    
    if len(projects) == 0:
         return [], {'plus': {}, 'minus': {}, 'net': {}, 'star': {}}, 0.0

    costs = matrix[:, cost_criterion_idx]
    
    # 1. Ranking (PROMETHEE II / Fluxo Star)
    project_scores = {p: s for p, s in zip(projects, phi_star)}
    sorted_projects = sorted(project_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 2. Seleção (PROMETHEE V C-ÓTIMO): Seleção sequencial gulosa no ranking
    portfolio, total_cost = [], 0.0
    for project, _ in sorted_projects:
        idx = projects.index(project)
        c = costs[idx]
        if total_cost + c <= budget_constraint:
            portfolio.append(project)
            total_cost += c
            
    return (
        portfolio,
        {'plus': dict(zip(projects, phi_plus)),
         'minus': dict(zip(projects, phi_minus)),
         'net': dict(zip(projects, phi_net)),
         'star': dict(zip(projects, phi_star))},
        total_cost
    )

def sensitivity_analysis_weights(
    matrix: np.ndarray, projects: List[str], criteria: List[str],
    optimization: Dict, original_weights: List[float], preference_functions: Dict,
    thresholds: Dict, budget_constraint: float, cost_criterion_idx: int,
    criterion_to_vary: str, n_steps: int = 20
) -> pd.DataFrame:
    """Realiza a Análise de Sensibilidade variando o peso de um critério específico."""
    original_idx = criteria.index(criterion_to_vary)
    
    # Gerar pontos para o peso do critério em análise (0.0 a 1.0)
    weight_range = np.linspace(0.0, 1.0, n_steps)
    results = []
    
    # Calcular a soma dos pesos originais dos outros critérios
    original_remaining_sum = sum(w for i, w in enumerate(original_weights) if i != original_idx)
    
    # Normalização segura dos pesos originais para distribuição proporcional
    if original_remaining_sum == 0.0:
        if len(criteria) > 1:
            # Se a soma dos demais é zero, assume distribuição uniforme para o peso restante
            original_weights_norm = [1.0] * len(criteria)
            if original_weights[original_idx] == 1.0:
                 original_weights_norm[original_idx] = 0.0
            original_remaining_sum = sum(original_weights_norm)
        else:
            original_weights_norm = original_weights
    else:
        original_weights_norm = original_weights
    
    for w_i in weight_range:
        # A soma dos pesos restantes deve ser 1 - w_i
        remaining_sum = 1.0 - w_i
        
        new_weights = [0.0] * len(criteria)
        new_weights[original_idx] = w_i
        
        # Distribuir o peso restante (1 - w_i) proporcionalmente aos outros critérios
        for i, w_orig in enumerate(original_weights_norm):
            if i != original_idx:
                if original_remaining_sum > 1e-6: # Evita divisão por zero
                    # Distribuição proporcional: (peso original / soma dos demais originais) * (1 - w_i)
                    new_weights[i] = (w_orig / original_remaining_sum) * remaining_sum
                elif len(criteria) > 1:
                    # Caso a soma dos outros pesos originais seja 0, divide-se o peso restante igualmente
                    new_weights[i] = remaining_sum / (len(criteria) - 1)
        
        # Recalcular os fluxos e o portfólio com os novos pesos
        
        if sum(new_weights) > 1e-6: # Evita pesos muito próximos de zero
            portfolio, scores, total_cost = promethee_v_c_otimo(
                matrix, projects, criteria, optimization, new_weights, 
                preference_functions, thresholds, budget_constraint, cost_criterion_idx
            )
            # Armazena o projeto com maior Fluxo Adaptado (Ranking PROMETHEE II)
            best_project = max(scores['star'].items(), key=lambda item: item[1])[0] if scores['star'] else None
        else:
            best_project = "Indefinido (Pesos nulos)"
            portfolio = []

        results.append({
            "weight": w_i,
            "best_project": best_project,
            "portfolio": ", ".join(portfolio)
        })

    return pd.DataFrame(results)

# =============================
# CATÁLOGO DE CRITÉRIOS
# =============================
predefined_criteria = {
    "Eficiência (custo-efetividade)": {
        "type": "Quantitativo (min)",
        "definition": "Relação entre o custo total do projeto e o número de pessoas ou unidades beneficiadas, visando o menor gasto por unidade de benefício.",
        "metric": "R$/beneficiário"
    },
    "Eficácia": {
        "type": "Quantitativo (max)",
        "definition": "Grau em que os objetivos específicos e metas quantificáveis do projeto foram atingidos no período estabelecido",
        "metric": "Porcentagem"
    },
    "Resultado": {
        "type": "Quantitativo (max)",
        "definition": "Entrega final e tangível de bens ou serviços gerados pelo projeto (Ex: número de casas construídas, número de alunos formados).",
        "metric": "Número de unidades entregues"
    },
    "Impacto": {
        "type": "Qualitativo (max)",
        "definition": "Efeitos de longo prazo, desejados ou não, que o projeto gera na vida dos beneficiários e nas estruturas sociais, após sua conclusão.",
        "metric": "Likert de 1 a 5 (1-muito baixo, 5-muito alto)",
        "scale": [
            {"Escala": "1 - Totalmente Discordo", "Descrição": "O projeto não gera impacto social significativo."},
            {"Escala": "2 - Discordo Parcialmente", "Descrição": "O impacto é mínimo e pouco perceptível."},
            {"Escala": "3 - Neutro", "Descrição": "O impacto é moderado, mas não transformador."},
            {"Escala": "4 - Concordo Parcialmente", "Descrição": "O projeto gera impacto positivo e relevante."},
            {"Escala": "5 - Totalmente Concordo", "Descrição": "O impacto é profundo e transformador na comunidade."}
        ]
    },
    "Importância do problema social": {
        "type": "Qualitativo (max)",
        "definition": "Relevância e urgência do problema que o projeto busca solucionar.",
        "metric": "Escala Likert de 1 a 5 (1-muito baixo, 5-muito alto)",
        "scale": [
            {"Escala": "1 - Totalmente Discordo", "Descrição": "O problema social abordado na pesquisa não tem importância social ou relevância."},
            {"Escala": "2 - Discordo Parcialmente", "Descrição": "O problema social tem pouca importância social ou sua relevância é questionável."},
            {"Escala": "3 - Neutro", "Descrição": "Não há clareza suficiente para determinar a importância social do problema, ou ele tem uma importância social moderada."},
            {"Escala": "4 - Concordo Parcialmente", "Descrição": "O problema social abordado é relevante e tem alguma importância social."},
            {"Escala": "5 - Totalmente Concordo", "Descrição": "O problema social é de extrema importância e relevância social, justificando plenamente a pesquisa."}
        ]
    },
    "Escalabilidade": {"type": "Quantitativo (max)", "definition": "Potencial de expandir o projeto para atingir mais pessoas ou áreas.", "metric": "Unidade"},
    "Custo-benefício": {"type": "Quantitativo (min)", "definition": "Relação entre os custos do projeto e os benefícios sociais gerados.", "metric": "Razão monetária"},
    "Sustentabilidade social": {
        "type": "Quantitativo (max)",
        "definition": "Capacidade do projeto de ter continuidade e manter seus benefícios sociais após o término",
        "metric": "Porcentagem"
    },
    "Equidade social": {
        "type": "Quantitativo/qualitativo(max)",
        "definition": "Extensão em que o projeto garante acesso justo e igualitário aos seus benefícios, especialmente a grupos vulneráveis.",
        "metric": "Índice de inclusão (0-100) Escala Likert de 1 a 5 (1-muito baixo, 5-muito alto)",
        "scale": [
            {"Escala": "1 - Totalmente Discordo", "Descrição": "O projeto ou solução proposta não contribui para a equidade social e pode até agravar desigualdades existentes."},
            {"Escala": "2 - Discordo Parcialmente", "Descrição": "O projeto tem pouca ou nenhuma consideração pela equidade social, e seus benefícios não são distribuídos de forma justa."},
            {"Escala": "3 - Neutro", "Descrição": "O projeto tem um impacto neutro ou indefinido sobre a equidade social. Não promove nem prejudica a distribuição justa de recursos e oportunidades."},
            {"Escala": "4 - Concordo Parcialmente", "Descrição": "O projeto considera a equidade social e se esforça para distribuir os benefícios de forma mais justa, embora possa haver espaço para melhorias."},
            {"Escala": "5 - Totalmente Concordo", "Descrição": "O projeto é fundamental para a promoção da equidade social, garantindo que os benefícios e oportunidades sejam distribuídos de forma justa e equitativa, especialmente para grupos vulneráveis."}
        ]
    },
    "Replicabilidade": {
        "type": "Quantitativo/qualitativo(max)",
        "definition": "Possibilidade de o projeto ser implementado em outros contextos.",
        "metric": "Unidade/ Escala Likert de 1 a 5 (1-muito baixo, 5-muito alto)",
        "scale": [
            {"Escala": "1 - Totalmente Discordo", "Descrição": "O projeto não pode ser replicado em outros contextos."},
            {"Escala": "2 - Discordo Parcialmente", "Descrição": "A replicabilidade é baixa e limitada."},
            {"Escala": "3 - Neutro", "Descrição": "A replicabilidade é moderada."},
            {"Escala": "4 - Concordo Parcialmente", "Descrição": "O projeto pode ser replicado com adaptações."},
            {"Escala": "5 - Totalmente Concordo", "Descrição": "O projeto é altamente replicável em diversos contextos."}
        ]
    },
    "Engajamento local": {
        "type": "Qualitativo (max)",
        "definition": "Participação ativa da comunidade, consultas públicas e benefícios sociais.",
        "metric": "Fuzzy/ Escala Likert de 1 a 5 (1-muito baixo, 5-muito alto)",
        "scale": [
            {"Escala": "1 - Totalmente Discordo", "Descrição": "O projeto não promove o engajamento local, excluindo a participação da comunidade."},
            {"Escala": "2 - Discordo Parcialmente", "Descrição": "O projeto tem baixo engajamento local e a participação da comunidade é mínima."},
            {"Escala": "3 - Neutro", "Descrição": "O projeto tem um engajamento local moderado."},
            {"Escala": "4 - Concordo Parcialmente", "Descrição": "O projeto promove o engajamento local e a participação da comunidade em suas atividades."},
            {"Escala": "5 - Totalmente Concordo", "Descrição": "O projeto é exemplar em seu engajamento local, com um alto nível de participação e autonomia da comunidade em todas as etapas."}
        ]
    }
}

# =============================
# FUNÇÃO: LOGO + TÍTULO
# =============================
def render_header(subtitle: str):
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        logo_path = "logo-PDMSPS.png" 
        if os.path.exists(logo_path):
            st.image(logo_path, width=350)
        else:
            st.image(
                "https://via.placeholder.com/350x200/003087/FFFFFF?text=PDM+PROJETOS",
                width=350
            )
    with col_title:
        st.markdown(
            "<h1 style='margin: 0; padding-top: 60px; color: #003087; line-height: 1.2;'>"
            "Processo de Decisão Multicritério para Seleção de Projetos Sociais"
            "</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='margin: 0; font-size: 18px; color: #555;'>{subtitle}</p>",
            unsafe_allow_html=True
        )
    st.markdown("<hr style='border: 1.5px solid #003087; margin: 30px 0;'>", unsafe_allow_html=True)

# =============================
# SALVAR / CARREGAR SESSÃO
# =============================
def save_session_state():
    state = {
        "projects": st.session_state.get("projects", []),
        "decision_makers": st.session_state.get("decision_makers", []),
        "analysts": st.session_state.get("analysts", []),
        "experts": st.session_state.get("experts", []),
        "selected_criteria": st.session_state.get("selected_criteria", []),
        "crit_states": st.session_state.get("crit_states", {}),
        "matrix_df": st.session_state.get("cons_matrix", pd.DataFrame()).to_dict(orient="split"), # Salvando 'cons_matrix'
        "cost_criterion": st.session_state.get("cost_criterion", ""),
        "budget_constraint": st.session_state.get("budget_constraint", 0.0),
        "seed": st.session_state.get("seed", 42)
    }
    return json.dumps(state, indent=2, ensure_ascii=False).encode('utf-8')

def load_session_state(uploaded_file):
    if uploaded_file:
        try:
            state = json.load(uploaded_file)
            st.session_state.projects = state.get("projects", [])
            st.session_state.decision_makers = state.get("decision_makers", [])
            st.session_state.analysts = state.get("analysts", [])
            st.session_state.experts = state.get("experts", [])
            st.session_state.selected_criteria = state.get("selected_criteria", [])
            st.session_state.crit_states = state.get("crit_states", {})
            df_data = state.get("matrix_df", {})
            if df_data:
                # Carregando a matriz de volta
                loaded_df = pd.DataFrame(**df_data)
                st.session_state.cons_matrix = loaded_df
                # Reiniciando a data editor key para forçar atualização
                st.session_state['cons_matrix_key'] = np.random.randint(0, 100000)
            
            st.session_state.cost_criterion = state.get("cost_criterion", "")
            st.session_state.budget_constraint = state.get("budget_constraint", 0.0)
            st.session_state.seed = state.get("seed", 42)
            st.success("Sessão carregada com sucesso!")
            st.rerun()
        except Exception as e:
            st.error(f"Erro ao carregar: {e}")

# =============================
# PÁGINAS
# =============================
def page_home():
    render_header("Decision Support System (DSS) com PROMETHEE V C-ÓTIMO")
    
    # === EQUIPE ===
    st.markdown("### Equipe")
    st.markdown(
        """
        <div style="text-align: center; padding: 15px; background-color: #f8f9fa; border-radius: 10px; margin: 20px 0;">
            <p style="font-size: 19px; font-weight: bold; margin: 8px 0; color: #003087;">Prof. Drª. Luciana Hazin Alencar</p>
            <p style="font-size: 17px; font-weight: bold; margin: 8px 0; color: #000;">Gabriel Mendes de Souza</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    st.markdown("### **Instituições Parceiras**")
    logo_cols = st.columns(3)
    logos = [
        {"local": "logo-ufpe.png", "fallback": "https://via.placeholder.com/150x100/003087/FFFFFF?text=UFPE", "label": "Universidade Federal de Pernambuco (UFPE)"},
        {"local": "logo-departamento.jpg", "fallback": "https://via.placeholder.com/150x100/003087/FFFFFF?text=DEPROD", "label": "Departamento de Engenharia de Produção"},
        {"local": "logo-PMD.png", "fallback": "https://via.placeholder.com/150x100/003087/FFFFFF?text=PMD", "label": "PMD - Gestão e Desenvolvimento de Projetos"}
    ]
    for i, logo in enumerate(logos):
        with logo_cols[i]:
            if os.path.exists(logo["local"]):
                st.image(logo["local"], width=150, caption=logo["label"])
            else:
                st.image(logo["fallback"], width=150, caption=logo["label"])
    st.markdown("---")
    
    # === SOBRE O SISTEMA ===
    st.markdown("""
**Sobre o sistema**
* Desenvolvido no Grupo de Pesquisa de Gestão e Desenvolvimento de Projetos (PMD) do Departamento de Engenharia de Produção da Universidade Federal de Pernambuco (UFPE).
* Integra critérios **Sociais e Econômicos**.
* Permite configurar **funções de preferência**, **limiares ($q$, $p$)**, **direção (Max/Min)** e **pesos** por critério.
* **Salvar/Carregar sessão** com um clique.
* Inclui **Análise de Sensibilidade** para os pesos dos critérios.
    """)

def page_promethee_v():
    render_header("PROMETHEE V C-ÓTIMO — Avaliação Multicritério")
    
    # Inicialização de chaves de estado para evitar KeyErrors
    if "projects" not in st.session_state: st.session_state.projects = ["Projeto A", "Projeto B", "Projeto C"]
    if "crit_states" not in st.session_state: st.session_state.crit_states = {}
    if "selected_criteria" not in st.session_state: st.session_state.selected_criteria = []
    if "cons_matrix" not in st.session_state: st.session_state.cons_matrix = pd.DataFrame()
    if "cost_criterion" not in st.session_state: st.session_state.cost_criterion = ""
    if "budget_constraint" not in st.session_state: st.session_state.budget_constraint = 0.0
    if "seed" not in st.session_state: st.session_state.seed = 42
    if 'cons_matrix_key' not in st.session_state: st.session_state['cons_matrix_key'] = 0

    # === SALVAR / CARREGAR ===
    col_save, col_load = st.columns([1, 1])
    with col_save:
        st.download_button(
            label="**Salvar Sessão (.json)**",
            data=save_session_state(),
            file_name="sessao_promethee.json",
            mime="application/json",
            use_container_width=True
        )
    with col_load:
        uploaded = st.file_uploader("**Carregar Sessão**", type="json", label_visibility="collapsed")
        if uploaded:
            load_session_state(uploaded)

    # === PARTICIPANTES E PROJETOS ===
    st.subheader("**Participantes e Projetos**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### **Atores**")
        decision_makers = [s.strip() for s in st.text_input("Decisores", value="DM1, DM2, DM3", key="decisores_input").split(',') if s.strip()]
        analysts = [s.strip() for s in st.text_input("Analistas", value="Ana Engenheira, Bruno Economista", key="analistas_input").split(',') if s.strip()]
        experts = [s.strip() for s in st.text_input("Especialistas", value="Carlos Sociólogo, Maria Médica", key="experts_input").split(',') if s.strip()]
    
    with col2:
        st.markdown("### **Projetos Sociais**")
        with st.container(border=True):
            for i, proj in enumerate(st.session_state.projects):
                col1_p, col2_p = st.columns([5, 1])
                with col1_p:
                    new_name = st.text_input(
                        f"Projeto {i+1}",
                        value=proj,
                        key=f"proj_input_{i}",
                        label_visibility="collapsed"
                    )
                
                # Atualiza o nome, evitando nome vazio
                st.session_state.projects[i] = new_name.strip() or f"Projeto {i+1}"
                
                with col2_p:
                    # O botão de remoção precisa de uma lógica para remover o item da lista
                    if st.button("X", key=f"remove_proj_{i}", type="secondary", use_container_width=True):
                        st.session_state.projects.pop(i)
                        # Limpa o input se remover (opcional, mas limpa o estado)
                        if f"proj_input_{i}" in st.session_state:
                             del st.session_state[f"proj_input_{i}"]
                        st.rerun()
            
            if st.button("Adicionar Projeto", use_container_width=True):
                st.session_state.projects.append(f"Projeto {len(st.session_state.projects)+1}")
                st.rerun()
        
        projects = [p for p in st.session_state.projects if p.strip()]
        st.session_state.projects = projects

    # === RESUMO (Salva os participantes para o 'save_session_state' funcionar) ===
    st.session_state.decision_makers = decision_makers
    st.session_state.analysts = analysts
    st.session_state.experts = experts
    with st.expander("Resumo dos Participantes e Projetos", expanded=True):
        st.write("**Decisores:**", ", ".join(decision_makers))
        st.write("**Analistas:**", ", ".join(analysts))
        st.write("**Especialistas:**", ", ".join(experts))
        st.write("**Projetos:**", ", ".join(projects) if projects else "(nenhum)")

    # === CRITÉRIOS ===
    st.subheader("1) Critérios e Configuração por Critério")
    all_criteria = list(predefined_criteria.keys())
    
    # --- CORREÇÃO APLICADA AQUI ---
    
    # 1. Filtra os critérios da sessão para incluir apenas aqueles que estão no catálogo
    if st.session_state.selected_criteria:
        st.session_state.selected_criteria = [
            c for c in st.session_state.selected_criteria if c in all_criteria
        ]

    # 2. Define o padrão de forma segura
    default_criteria = st.session_state.selected_criteria if st.session_state.selected_criteria else [
        "Eficiência (custo-efetividade)", "Eficácia", "Importância do problema social"
    ]
    
    selected = st.multiselect(
        "Escolha critérios do catálogo:", options=all_criteria,
        default=default_criteria, # Usando a lista de critérios válidos ou o padrão.
        key="criteria_multiselect"
    )
    
    # --- FIM DA CORREÇÃO ---
    
    custom_text = st.text_input("Adicionar critérios personalizados (separe por vírgula)", key="custom_crit_input")
    if custom_text.strip():
        # ATENÇÃO: Os customizados não estão no `all_criteria` (catálogo). 
        # Eles são adicionados aqui e salvos na `selected_criteria` para persistência.
        selected += [c.strip() for c in custom_text.split(',') if c.strip()]
    
    # Atualiza a lista de critérios selecionados na sessão
    st.session_state.selected_criteria = selected 
    
    if not selected:
        st.warning("Selecione pelo menos um critério.")
        return

    # Inicializa/atualiza o estado dos critérios
    default_opt = optimize_criteria_default(selected)
    for c in selected:
        if c not in st.session_state.crit_states:
            st.session_state.crit_states[c] = {
                "direction": default_opt.get(c, "Maximize"),
                "func": 'u',
                "q": 0.0,
                "p": 1.0,
                "weight": round(1.0/len(selected), 4) if len(selected) > 0 else 1.0
            }
    for c in list(st.session_state.crit_states.keys()):
        if c not in selected:
            st.session_state.crit_states.pop(c)

    # Exibição e Configuração dos Critérios
    for c in selected:
        st.markdown("---")
        st.markdown(f"### **{c}**")
        
        # Define as colunas de info e config
        col_info, col_config = st.columns([3, 1])
        
        with col_info:
            if c in predefined_criteria:
                info = predefined_criteria[c]
                st.caption(f"{info['type']} - {info['metric']}")
                with st.expander("**Definição**", expanded=False):
                    st.write(info['definition'])
                if 'scale' in info:
                    with st.expander(f"**Escala Likert (1-5)**", expanded=False): # Mudei para False para não poluir
                        scale_df = pd.DataFrame(info['scale'])
                        st.table(scale_df.style.set_properties(**{'text-align': 'left', 'font-size': '12px'}))
                        st.caption("Fonte: O autor (2025)")
            else:
                st.caption("Critério Customizado")
                
        with col_config:
            st.markdown("**Configurações**")
            # Usa o estado atual para definir o valor padrão (index)
            current_dir_index = 0 if st.session_state.crit_states[c]["direction"] == "Maximize" else 1
            direction = st.selectbox("Direção", ["Maximize", "Minimize"],
                                     index=current_dir_index,
                                     key=f"dir_{c}")
            st.session_state.crit_states[c]["direction"] = direction
            
            current_func_index = {"u":0,"v":1,"l":2}.get(st.session_state.crit_states[c]["func"], 0)
            func_label = st.selectbox("Função", ["usual", "v-shape", "linear"],
                                     index=current_func_index,
                                     key=f"fun_{c}")
            st.session_state.crit_states[c]["func"] = {"usual": 'u', "v-shape": 'v', "linear": 'l'}[func_label]
            
            if st.session_state.crit_states[c]["func"] != 'u':
                q = st.number_input("$q$ (Indiferença)", value=float(st.session_state.crit_states[c]["q"]), key=f"q_{c}", step=0.1)
                p = st.number_input("$p$ (Preferência)", value=float(st.session_state.crit_states[c]["p"]), min_value=0.0, key=f"p_{c}", step=0.1)
                st.session_state.crit_states[c]["q"] = q
                st.session_state.crit_states[c]["p"] = p
                if p <= q:
                    st.error(f"Para '{c}', $p$ deve ser maior que $q$.")
            else:
                st.session_state.crit_states[c]["q"] = 0.0
                st.session_state.crit_states[c]["p"] = 0.0

    # === PESOS ===
    st.markdown("---")
    st.subheader("2) Pesos")
    weights_cols = st.columns(min(4, len(selected)))
    for idx, c in enumerate(selected):
        with weights_cols[idx % len(weights_cols)]:
            st.session_state.crit_states[c]["weight"] = float(st.slider(
                f"Peso — {c}", 0.0, 1.0,
                value=float(st.session_state.crit_states[c]["weight"]),
                step=0.01, key=f"w_{c}"
            ))
    
    sum_w = sum(st.session_state.crit_states[c]["weight"] for c in selected)
    col_norm_a, col_norm_b = st.columns([1, 3])
    with col_norm_a:
        if st.button("Normalizar pesos (soma=1)", key="normalize_btn"):
            if sum_w == 0:
                st.warning("Soma dos pesos é 0. Defina um peso inicial.")
            else:
                for c in selected:
                    st.session_state.crit_states[c]["weight"] = st.session_state.crit_states[c]["weight"] / sum_w
                st.success("Pesos normalizados! (Recalculando)")
                st.rerun()
    with col_norm_b:
        st.info(f"Soma dos pesos: **{sum_w:.3f}**")

    # === MATRIZ E ORÇAMENTO ===
    st.subheader("3) Matriz de Consequência")
    st.session_state.seed = st.number_input("Seed (opcional para valores iniciais)", min_value=0, value=st.session_state.seed, step=1, key="seed_input")
    
    # Se a matriz não foi carregada ou está vazia, cria uma nova
    if st.session_state.cons_matrix.empty or list(st.session_state.cons_matrix.columns) != selected or list(st.session_state.cons_matrix.index) != projects:
        matrix_df = build_consequence_matrix(projects, selected, seed=st.session_state.seed)
        st.session_state.cons_matrix = matrix_df

    # Se a matriz foi carregada, usa a carregada e garante que a chave muda
    edit_df = st.data_editor(st.session_state.cons_matrix, 
                             use_container_width=True, 
                             num_rows="dynamic", 
                             key=f"cons_matrix_{st.session_state['cons_matrix_key']}")
    
    # Atualiza a matriz editada na sessão
    st.session_state.cons_matrix = edit_df 
    
    # Configuração de Custo/Orçamento
    default_cost_idx = selected.index(st.session_state.cost_criterion) if st.session_state.cost_criterion in selected else 0
    cost_criterion = st.selectbox("Critério de CUSTO", options=selected, index=default_cost_idx, key="cost_crit_select")
    st.session_state.cost_criterion = cost_criterion
    
    cost_idx = selected.index(cost_criterion)
    
    # Valor padrão do orçamento: metade da soma dos custos
    default_budget = float(edit_df.iloc[:, cost_idx].sum()/2) if len(edit_df) > 0 else 0.0
    
    # Se for a primeira execução ou for carregado 0, usa o default
    if st.session_state.budget_constraint <= 0.0:
         st.session_state.budget_constraint = default_budget

    budget_constraint = st.number_input("Restrição orçamentária", min_value=0.0,
                                         value=float(st.session_state.budget_constraint), 
                                         step=1.0, key="budget_input")
    st.session_state.budget_constraint = budget_constraint
    

    # === EXECUTAR ===
    st.markdown("---")
    if st.button("**Executar PROMETHEE V-C-ÓTIMO**", key="run_promethee_btn", type="primary"):
        invalid_thresholds = [c for c in selected if st.session_state.crit_states[c]["func"] != 'u' and
                              st.session_state.crit_states[c]['p'] <= st.session_state.crit_states[c]['q']]
        
        if invalid_thresholds:
            st.error(f"Critério(s) com $p \leq q$: {', '.join(invalid_thresholds)}")
            return
            
        opt_choices = {c: st.session_state.crit_states[c]["direction"] for c in selected}
        preference_functions = {c: st.session_state.crit_states[c]["func"] for c in selected}
        thresholds = {c: {"q": st.session_state.crit_states[c]["q"], "p": st.session_state.crit_states[c]["p"]}
                      for c in selected}
        weight_values = [st.session_state.crit_states[c]["weight"] for c in selected]
        
        if sum(weight_values) == 0:
            st.error("Soma dos pesos não pode ser 0.")
            return
        
        # Usa a matriz atualizada do data_editor
        matrix = edit_df.to_numpy(dtype=float) 
        
        portfolio, scores, total_cost = promethee_v_c_otimo(matrix, projects, selected, opt_choices,
                                                             weight_values, preference_functions, thresholds,
                                                             float(budget_constraint), cost_idx)
        
        st.session_state.last_results = {
            "portfolio": portfolio,
            "scores": scores,
            "total_cost": total_cost,
            "weights": weight_values,
            "matrix": matrix,
            "opt_choices": opt_choices,
            "preference_functions": preference_functions,
            "thresholds": thresholds,
            "cost_idx": cost_idx
        }
        st.success("**Avaliação concluída!** Os resultados estão abaixo.")
        
    if st.session_state.get('last_results'):
        results = st.session_state['last_results']
        portfolio = results['portfolio']
        scores = results['scores']
        weight_values = results['weights']
        matrix = results['matrix']
        opt_choices = results['opt_choices']
        preference_functions = results['preference_functions']
        thresholds = results['thresholds']
        total_cost = results['total_cost']
        
        # === RESULTADO PRINCIPAL ===
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("🎯 **Portfólio Recomendado**")
            st.markdown(f"**Projetos Selecionados:** {', '.join(portfolio) if portfolio else '(nenhum)'}")
            st.info(f"O portfólio maximiza o $\\varphi^*$ (Fluxo Adaptado) total, respeitando o orçamento de **R${budget_constraint:,.2f}**.")
        with col2:
            st.metric("Custo Total do Portfólio", f"R${total_cost:,.2f}")
            st.metric("Critério de Custo", cost_criterion)

        # === FLUXOS DETALHADOS ===
        st.subheader("📊 Fluxos Detalhados (PROMETHEE V C-ÓTIMO)")
        
        # Recalcula os fluxos para ter o T e os valores
        phi_plus, phi_minus, phi_net, phi_star = compute_preference_flows(
             matrix, selected, opt_choices, weight_values, preference_functions, thresholds
        )
        T = max(0.0, -np.min(phi_net)) if len(phi_net) > 0 else 0.0
        
        fluxos_df = pd.DataFrame({
            'Projeto': projects,
            'Fluxo Positivo (φ⁺)': [round(x, 4) for x in phi_plus],
            'Fluxo Negativo (φ⁻)': [round(x, 4) for x in phi_minus],
            'Fluxo Líquido (φ)': [round(x, 4) for x in phi_net],
            'Fluxo Adaptado (φ*)': [round(x, 4) for x in phi_star]
        }).sort_values("Fluxo Adaptado (φ*)", ascending=False).reset_index(drop=True)
        fluxos_df.index = fluxos_df.index + 1
        st.dataframe(fluxos_df, use_container_width=True)

        # Gráfico
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(projects))
        width = 0.18
        ax.bar(x - 1.5*width, phi_plus, width, label='φ⁺ (Positivo)', color='green', alpha=0.8)
        ax.bar(x - 0.5*width, phi_minus, width, label='φ⁻ (Negativo)', color='red', alpha=0.8)
        ax.bar(x + 0.5*width, phi_net, width, label='φ (Líquido)', color='blue', alpha=0.8)
        ax.bar(x + 1.5*width, phi_star, width, label='φ* (Adaptado)', color='purple', alpha=0.9)
        ax.set_xlabel('Projetos')
        ax.set_ylabel('Fluxos')
        ax.set_title('Fluxos PROMETHEE V C-ÓTIMO')
        ax.set_xticks(x)
        ax.set_xticklabels(projects, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        st.caption(
            f"**Constante de Ajuste $T = {T:.4f}$**"
        )
        
        # Exportar
        st.markdown("---")
        col_csv_1, col_csv_2 = st.columns(2)
        with col_csv_1:
            csv = fluxos_df.to_csv(index=True, header=True).encode('utf-8')
            st.download_button("Baixar Fluxos Completos (CSV)", data=csv, file_name="fluxos_promethee_v_cotimo.csv", mime="text/csv", use_container_width=True)
        with col_csv_2:
            portfolio_df = pd.DataFrame({'Projeto': portfolio, 'Custo': [edit_df.loc[p, cost_criterion] for p in portfolio]})
            st.download_button("Baixar Portfólio (CSV)", data=portfolio_df.to_csv(index=False).encode('utf-8'),
                                file_name="portfolio_selecionado.csv", mime="text/csv", use_container_width=True)

def page_sensitivity():
    render_header("Análise de Sensibilidade 📊 — Variação de Pesos")

    # Verifica se a avaliação principal foi executada
    if st.session_state.get("last_results") is None:
        st.warning("⚠️ Primeiro, **execute a avaliação principal** na página 'PROMETHEE V-C-ÓTIMO' para carregar os parâmetros necessários.")
        return
        
    results = st.session_state['last_results']
    selected = st.session_state.selected_criteria
    projects = st.session_state.projects
    
    st.subheader("Configuração da Análise de Sensibilidade")
    
    # O critério a ter seu peso variado
    crit_to_vary = st.selectbox(
        "Selecione o Critério para Análise de Sensibilidade (variar peso de 0 a 1):", 
        options=selected, 
        key="crit_to_vary_sens"
    )
    
    # Número de passos para a variação
    n_steps = st.slider(
        "Número de Pontos de Variação do Peso (Resolução)", 
        min_value=10, 
        max_value=100, 
        value=20, 
        step=10,
        key="n_steps_sens"
    )
    
    st.markdown("---")
    if st.button("Executar Análise de Sensibilidade", type="primary", use_container_width=True, key="run_sensitivity_btn"):
        
        # Recuperar parâmetros do último cálculo
        matrix = results['matrix']
        opt_choices = results['opt_choices']
        original_weight_values = results['weights']
        preference_functions = results['preference_functions']
        thresholds = results['thresholds']
        cost_idx = results['cost_idx']
        budget_constraint = st.session_state.budget_constraint # Pega o orçamento da sessão
        
        if sum(original_weight_values) == 0:
            st.error("A soma dos pesos originais é zero. Normalize os pesos antes de rodar a sensibilidade.")
            st.session_state.sensitivity_results = None
            return

        # Execução da análise
        results_df = sensitivity_analysis_weights(
            matrix, projects, selected, opt_choices, original_weight_values, 
            preference_functions, thresholds, float(budget_constraint), cost_idx, 
            crit_to_vary, n_steps
        )
        st.session_state.sensitivity_results = results_df
        st.success(f"Análise de sensibilidade para **{crit_to_vary}** concluída com **{n_steps}** passos.")
        st.rerun()

    if st.session_state.get("sensitivity_results") is not None:
        results_df = st.session_state.sensitivity_results
        
        st.markdown("### Resultados Detalhados da Sensibilidade")
        st.info(f"O peso de **{crit_to_vary}** varia de 0 a 1, e os demais pesos são **ajustados proporcionalmente** para que a soma seja sempre $1.0$.")

        # Tabela
        results_df.rename(columns={"weight": f"Peso de {crit_to_vary}", 
                                   "best_project": "Melhor Projeto (φ* Máximo)", 
                                   "portfolio": "Portfólio Selecionado (C-Ótimo)"}, inplace=True)
        st.dataframe(results_df, use_container_width=True)
        
        # Gráfico (Melhor Projeto)
        st.markdown("### Variação do Melhor Projeto em Relação ao Peso")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        unique_projects = results_df['Melhor Projeto (φ* Máximo)'].unique()
        proj_map = {proj: i for i, proj in enumerate(unique_projects)}
        results_df['best_proj_val'] = results_df['Melhor Projeto (φ* Máximo)'].map(proj_map)
        
        ax.plot(results_df[f"Peso de {crit_to_vary}"], results_df['best_proj_val'], 'o-', label="Melhor Projeto", color='#003087')
        ax.set_yticks(list(proj_map.values()))
        ax.set_yticklabels(list(proj_map.keys()))
        ax.set_xlabel(f"Peso do Critério: {crit_to_vary}")
        ax.set_ylabel("Projeto com o maior Fluxo Adaptado (φ*)")
        ax.set_title(f"Sensibilidade do Ranking em Relação ao Peso de {crit_to_vary}")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.caption("Qualquer mudança vertical no gráfico indica um **ponto crítico** onde a classificação do melhor projeto (baseada no $\\varphi^*$) se inverte devido à variação do peso.")
        
        # Exportar
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("Baixar Resultados da Sensibilidade (CSV)", data=csv, file_name="sensibilidade_pesos.csv", mime="text/csv", use_container_width=True)

# =============================
# ROTEAMENTO
# =============================
menu = ["Home", "PROMETHEE V-C-ÓTIMO", "Análise de Sensibilidade"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    page_home()
elif choice == "PROMETHEE V-C-ÓTIMO":
    page_promethee_v()
elif choice == "Análise de Sensibilidade":
    page_sensitivity()

st.sidebar.caption("© PDMSPS — Sistema de Apoio à Decisão")
