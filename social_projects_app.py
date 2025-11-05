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

    for i in range(n_projects):
        for j in range(n_projects):
            if i == j:
                continue
            sum_pref = 0.0
            for k, c in enumerate(criteria):
                d = matrix[i, k] - matrix[j, k] if optimization[c] == "Maximize" else matrix[j, k] - matrix[i, k]
                func = preference_functions[c]
                q = thresholds.get(c, {}).get('q', 0)
                p = thresholds.get(c, {}).get('p', 1.0)
                if d <= 0:
                    pref = 0.0
                elif func == 'u':
                    pref = 1.0
                elif func == 'v':
                    pref = min(1.0, d / p)
                elif func == 'l':
                    pref = 0.0 if d <= q else min(1.0, (d - q) / (p - q))
                else:
                    pref = 0.0
                sum_pref += weights[k] * pref
            pref_overall = sum_pref / (n_projects - 1)
            phi_plus[i] += pref_overall
            phi_minus[j] += pref_overall
    phi_net = phi_plus - phi_minus
    T = max(0.0, -np.min(phi_net))
    phi_star = phi_net + T
    return phi_plus, phi_minus, phi_net, phi_star

def promethee_v_c_otimo(matrix: np.ndarray, projects: List[str], criteria: List[str],
                       optimization: Dict, weights: List[float], preference_functions: Dict,
                       thresholds: Dict, budget_constraint: float, cost_criterion_idx: int):
    phi_plus, phi_minus, phi_net, phi_star = compute_preference_flows(
        matrix, criteria, optimization, weights, preference_functions, thresholds
    )
    costs = matrix[:, cost_criterion_idx]
    project_scores = {p: s for p, s in zip(projects, phi_star)}
    sorted_projects = sorted(project_scores.items(), key=lambda x: x[1], reverse=True)
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

# =============================
# CATÁLOGO DE CRITÉRIOS (CORRIGIDO: coastline → metric)
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
# FUNÇÃO: LOGO + TÍTULO (USANDO SEUS NOMES)
# =============================
def render_header(subtitle: str):
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        logo_path = "logo-PDMSPS.png"  # ← Seu nome real
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
        "matrix_df": st.session_state.get("matrix_df", pd.DataFrame()).to_dict(orient="split"),
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
                st.session_state.matrix_df = pd.DataFrame(**df_data)
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
    st.markdown("### Equipe")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Prof. Drª. Luciana Hazin Alencar**")
    with cols[1]:
        st.markdown("**Gabriel Mendes de Souza**")
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
    
    # ← TEXTO CORRIGIDO COM BULLETS PERFEITOS
    st.markdown("""
**Sobre o sistema**
- Desenvolvido no Grupo de Pesquisa de Gestão e Desenvolvimento de Projetos (PMD) do Departamento de Engenharia de Produção da Universidade Federal de Pernambuco (UFPE).
- Integra critérios **Sociais e Econômicos**.
- Permite configurar **funções de preferência**, **limiares (q, p)**, **direção (Max/Min)** e **pesos** por critério.
- **Salvar/Carregar sessão** com um clique.
    """)

def page_promethee_v():
    render_header("PROMETHEE V C-ÓTIMO — Avaliação Multicritério")
    
    # === SALVAR / CARREGAR ===
    col_save, col_load = st.columns([1, 1])
    with col_save:
        if st.button("**Salvar Sessão (.json)**", use_container_width=True):
            data = save_session_state()
            st.download_button(
                label="Baixar Arquivo de Sessão",
                data=data,
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
        decision_makers = [s.strip() for s in st.text_input("Decisores", value="DM1, DM2, DM3").split(',') if s.strip()]
        analysts = [s.strip() for s in st.text_input("Analistas", value="Ana Engenheira, Bruno Economista").split(',') if s.strip()]
        experts = [s.strip() for s in st.text_input("Especialistas", value="Carlos Sociólogo, Maria Médica").split(',') if s.strip()]
   
    with col2:
        st.markdown("### **Projetos Sociais**")
        if "projects" not in st.session_state:
            st.session_state.projects = ["Projeto A", "Projeto B", "Projeto C"]
        with st.container():
            for i, proj in enumerate(st.session_state.projects):
                col1, col2 = st.columns([5, 1])
                with col1:
                    new_name = st.text_input(
                        f"Projeto {i+1}",
                        value=proj,
                        key=f"proj_input_{i}",
                        label_visibility="collapsed"
                    )
                    st.session_state.projects[i] = new_name.strip() or f"Projeto {i+1}"
                with col2:
                    if st.button("X", key=f"remove_proj_{i}", type="secondary"):
                        st.session_state.projects.pop(i)
                        st.rerun()
            if st.button("Adicionar Projeto", use_container_width=True):
                st.session_state.projects.append(f"Projeto {len(st.session_state.projects)+1}")
                st.rerun()
        projects = [p for p in st.session_state.projects if p.strip()]
        st.session_state.projects = projects

    # === RESUMO ===
    with st.expander("Resumo dos Participantes e Projetos", expanded=True):
        st.write("**Decisores:**", ", ".join(decision_makers))
        st.write("**Analistas:**", ", ".join(analysts))
        st.write("**Especialistas:**", ", ".join(experts))
        st.write("**Projetos:**", ", ".join(projects) if projects else "(nenhum)")

    # === CRITÉRIOS ===
    st.subheader("1) Critérios e Configuração por Critério")
    all_criteria = list(predefined_criteria.keys())
    selected = st.multiselect(
        "Escolha critérios do catálogo:", options=all_criteria,
        default=["Eficiência (custo-efetividade)", "Eficácia", "Importância do problema social"]
    )
    custom_text = st.text_input("Adicionar critérios personalizados (separe por vírgula)")
    if custom_text.strip():
        selected += [c.strip() for c in custom_text.split(',') if c.strip()]
    if not selected:
        st.warning("Selecione pelo menos um critério.")
        return

    if "crit_states" not in st.session_state:
        st.session_state.crit_states = {}
    default_opt = optimize_criteria_default(selected)
    for c in selected:
        if c not in st.session_state.crit_states:
            st.session_state.crit_states[c] = {
                "direction": default_opt.get(c, "Maximize"),
                "func": 'u',
                "q": 0.0,
                "p": 1.0,
                "weight": round(1.0/len(selected), 4)
            }
    for c in list(st.session_state.crit_states.keys()):
        if c not in selected:
            st.session_state.crit_states.pop(c)

    for c in selected:
        st.markdown("---")
        st.markdown(f"### **{c}**")
        if c in predefined_criteria:
            info = predefined_criteria[c]
            col_info, col_config = st.columns([3, 1])
            with col_info:
                st.caption(f"{info['type']} - {info['metric']}")
                with st.expander("**Definição**", expanded=False):
                    st.write(info['definition'])
                if 'scale' in info:
                    with st.expander(f"**Escala Likert (1-5)**", expanded=True):
                        scale_df = pd.DataFrame(info['scale'])
                        st.table(scale_df.style.set_properties(**{'text-align': 'left', 'font-size': '12px'}))
                        st.caption("Fonte: O autor (2025)")
        else:
            st.caption("Custom")
            col_info, col_config = st.columns([3, 1])
        with col_config:
            st.markdown("**Configurações**")
            direction = st.selectbox("Direção", ["Maximize", "Minimize"],
                                   index=0 if st.session_state.crit_states[c]["direction"]=="Maximize" else 1,
                                   key=f"dir_{c}")
            st.session_state.crit_states[c]["direction"] = direction
            func_label = st.selectbox("Função", ["usual", "v-shape", "linear"],
                                    index={"u":0,"v":1,"l":2}[st.session_state.crit_states[c]["func"]],
                                    key=f"fun_{c}")
            st.session_state.crit_states[c]["func"] = {"usual": 'u', "v-shape": 'v', "linear": 'l'}[func_label]
            if st.session_state.crit_states[c]["func"] != 'u':
                q = st.number_input("q (Indiferença)", value=float(st.session_state.crit_states[c]["q"]), key=f"q_{c}")
                p = st.number_input("p (Preferência)", value=float(st.session_state.crit_states[c]["p"]), min_value=0.0, key=f"p_{c}")
                st.session_state.crit_states[c]["q"] = q
                st.session_state.crit_states[c]["p"] = p
                if p <= q:
                    st.error(f"Para '{c}', p deve ser maior que q.")
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
        if st.button("Normalizar pesos (soma=1)"):
            if sum_w == 0:
                st.warning("Soma dos pesos é 0.")
            else:
                for c in selected:
                    st.session_state.crit_states[c]["weight"] = st.session_state.crit_states[c]["weight"] / sum_w
                st.success("Pesos normalizados!")
    with col_norm_b:
        st.info(f"Soma dos pesos: **{sum_w:.3f}**")

    # === MATRIZ E ORÇAMENTO ===
    st.subheader("3) Matriz de Consequência")
    seed = st.number_input("Seed (opcional)", min_value=0, value=42, step=1)
    matrix_df = build_consequence_matrix(projects, selected, seed=int(seed))
    edit_df = st.data_editor(matrix_df, use_container_width=True, num_rows="dynamic", key="cons_matrix")
    cost_criterion = st.selectbox("Critério de CUSTO", options=selected, index=0)
    cost_idx = selected.index(cost_criterion)
    budget_constraint = st.number_input("Restrição orçamentária", min_value=0.0,
                                      value=float(edit_df.iloc[:, cost_idx].sum()/2 if len(edit_df) else 0.0), step=1.0)

    # === EXECUTAR ===
    if st.button("**Executar PROMETHEE V-C-ÓTIMO**"):
        invalid_thresholds = [c for c in selected if st.session_state.crit_states[c]["func"] != 'u' and
                            st.session_state.crit_states[c]['p'] <= st.session_state.crit_states[c]['q']]
       
        if invalid_thresholds:
            st.error(f"Critério(s) com p ≤ q: {', '.join(invalid_thresholds)}")
            return
           
        opt_choices = {c: st.session_state.crit_states[c]["direction"] for c in selected}
        preference_functions = {c: st.session_state.crit_states[c]["func"] for c in selected}
        thresholds = {c: {"q": st.session_state.crit_states[c]["q"], "p": st.session_state.crit_states[c]["p"]}
                     for c in selected}
        weight_values = [st.session_state.crit_states[c]["weight"] for c in selected]
       
        if sum(weight_values) == 0:
            st.error("Soma dos pesos não pode ser 0.")
            return
        matrix = edit_df.to_numpy(dtype=float)
        portfolio, scores, total_cost = promethee_v_c_otimo(matrix, projects, selected, opt_choices,
                                                          weight_values, preference_functions, thresholds,
                                                          float(budget_constraint), cost_idx)
        st.success("**Avaliação concluída!**")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("**Portfólio Recomendado**")
            st.write(", ".join(portfolio) if portfolio else "(nenhum)")
            st.write(f"**Custo total:** {total_cost:.2f}")
        with col2:
            st.metric("Custo (critério)", cost_criterion)

        # === FLUXOS DETALHADOS ===
        st.subheader("Fluxos Detalhados (PROMETHEE V C-ÓTIMO)")
        phi_plus, phi_minus, phi_net, phi_star = compute_preference_flows(
            matrix, selected, opt_choices, weight_values, preference_functions, thresholds
        )
        T = max(0.0, -np.min(phi_net))
        fluxos_df = pd.DataFrame({
            'Projeto': projects,
            'Fluxo Positivo (φ⁺)': [round(x, 4) for x in phi_plus],
            'Fluxo Negativo (φ⁻)': [round(x, 4) for x in phi_minus],
            'Fluxo Líquido (φ)': [round(x, 4) for x in phi_net],
            'Fluxo Adaptado (φ*)': [round(x, 4) for x in phi_star]
        }).sort_values("Fluxo Adaptado (φ*)", ascending=False)
        st.dataframe(fluxos_df, use_container_width=True)

        # Gráfico (CORRIGIDO: set_xticklabels)
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(projects))
        width = 0.2
        ax.bar(x - 1.5*width, phi_plus, width, label='φ⁺ (Positivo)', color='green', alpha=0.8)
        ax.bar(x - 0.5*width, phi_minus, width, label='φ⁻ (Negativo)', color='red', alpha=0.8)
        ax.bar(x + 0.5*width, phi_net, width, label='φ (Líquido)', color='blue', alpha=0.8)
        ax.bar(x + 1.5*width, phi_star, width, label='φ* (Adaptado)', color='purple', alpha=0.9)
        ax.set_xlabel('Projetos')
        ax.set_ylabel('Fluxos')
        ax.set_title('Fluxos PROMETHEE V C-ÓTIMO')
        ax.set_xticks(x)
        ax.set_xticklabels(projects, rotation=45)  # ← CORRIGIDO!
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        st.caption(
            "• φ⁺: Quanto o projeto domina os outros\n"
            "• φ⁻: Quanto o projeto é dominado\n"
            "• φ = φ⁺ - φ⁻ (ranking PROMETHEE II)\n"
            "• **φ* = φ + T** → Fluxo Líquido Adaptado (T = max(0, -min(φ)))\n"
            "• **φ* ≥ 0** → usado no modelo de otimização (PROMETHEE V C-ÓTIMO)"
        )
        st.info(f"**Constante de Ajuste T = {T:.4f}** → Todos os φ* são não negativos")

        # Exportar
        csv = fluxos_df.to_csv(index=False).encode('utf-8')
        st.download_button("Baixar Fluxos Completos (CSV)", data=csv, file_name="fluxos_promethee_v_cotimo.csv", mime="text/csv")
        st.download_button("Baixar Portfólio (CSV)", data=pd.DataFrame({'Projeto': portfolio}).to_csv(index=False).encode('utf-8'),
                          file_name="portfolio.csv", mime="text/csv")

# =============================
# ROTEAMENTO
# =============================
menu = ["Home", "PROMETHEE V-C-ÓTIMO"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    page_home()
elif choice == "PROMETHEE V-C-ÓTIMO":
    page_promethee_v()

st.caption("© PDMSPS (Processo de Decisão Multicritério para Seleção de Projetos Sociais) — Sistema de Apoio à Decisão")
