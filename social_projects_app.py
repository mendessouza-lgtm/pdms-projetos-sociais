import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from typing import List, Dict, Tuple

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

def compute_preference_flows(matrix: np.ndarray, criteria: List[str], optimization: Dict, weights: List[float], preference_functions: Dict, thresholds: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Calcula:
    - φ⁺ (outflow)
    - φ⁻ (inflow)
    - φ (net flow)
    - φ* = φ + T  → FLUXO LÍQUIDO ADAPTADO (T = max(0, -min(φ)))
    """
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
    T = max(0.0, -np.min(phi_net))  # CONSTANTE DE AJUSTE (Vetschera & Almeida, 2012)
    phi_star = phi_net + T  # φ* ≥ 0 → usado na otimização

    return phi_plus, phi_minus, phi_net, phi_star, T

def promethee_v_c_otimo(matrix: np.ndarray, projects: List[str], criteria: List[str], optimization: Dict, weights: List[float], preference_functions: Dict, thresholds: Dict, budget_constraint: float, cost_criterion_idx: int):
    phi_plus, phi_minus, phi_net, phi_star, T = compute_preference_flows(
        matrix, criteria, optimization, weights, preference_functions, thresholds
    )
    costs = matrix[:, cost_criterion_idx]
    project_scores = {p: s for p, s in zip(projects, phi_star)}  # Usa φ*
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
        {
            'plus': dict(zip(projects, phi_plus)),
            'minus': dict(zip(projects, phi_minus)),
            'net': dict(zip(projects, phi_net)),
            'star': dict(zip(projects, phi_star)),
            'T': T
        },
        total_cost
    )

# =============================
# CATÁLOGO DE CRITÉRIOS (Vetschera & Almeida, 2012)
# =============================
predefined_criteria = {
    "Eficiência (custo-efetividade)": {
        "type": "Quantitativo (min)",
        "definition": "Relação entre o custo total do projeto e o número de pessoas ou unidades beneficiadas.",
        "metric": "R$/beneficiário"
    },
    "Eficácia": {
        "type": "Quantitativo (max)",
        "definition": "Grau em que os objetivos específicos foram atingidos.",
        "metric": "Porcentagem"
    },
    "Resultado": {
        "type": "Quantitativo (max)",
        "definition": "Entrega final e tangível (ex: casas construídas).",
        "metric": "Número de unidades"
    },
    "Impacto": {
        "type": "Qualitativo (max)",
        "definition": "Efeitos de longo prazo na comunidade.",
        "metric": "Likert 1-5",
        "scale": [
            {"Escala": "1", "Descrição": "Impacto muito baixo"},
            {"Escala": "2", "Descrição": "Impacto baixo"},
            {"Escala": "3", "Descrição": "Impacto moderado"},
            {"Escala": "4", "Descrição": "Impacto alto"},
            {"Escala": "5", "Descrição": "Impacto transformador"}
        ]
    },
    "Importância do problema social": {
        "type": "Qualitativo (max)",
        "definition": "Relevância e urgência do problema.",
        "metric": "Likert 1-5",
        "scale": [
            {"Escala": "1", "Descrição": "Baixa relevância"},
            {"Escala": "2", "Descrição": "Relevância questionável"},
            {"Escala": "3", "Descrição": "Relevância moderada"},
            {"Escala": "4", "Descrição": "Alta relevância"},
            {"Escala": "5", "Descrição": "Extrema relevância"}
        ]
    },
    "Escalabilidade": {"type": "Quantitativo (max)", "definition": "Potencial de expansão.", "metric": "Unidade"},
    "Custo-benefício": {"type": "Quantitativo (min)", "definition": "Relação custo/benefício.", "metric": "Razão"},
    "Sustentabilidade social": {"type": "Quantitativo (max)", "definition": "Continuidade após o término.", "metric": "Porcentagem"},
    "Equidade social": {
        "type": "Qualitativo (max)",
        "definition": "Acesso justo a grupos vulneráveis.",
        "metric": "Likert 1-5",
        "scale": [
            {"Escala": "1", "Descrição": "Sem equidade"},
            {"Escala": "2", "Descrição": "Baixa equidade"},
            {"Escala": "3", "Descrição": "Equidade moderada"},
            {"Escala": "4", "Descrição": "Alta equidade"},
            {"Escala": "5", "Descrição": "Equidade exemplar"}
        ]
    },
    "Replicabilidade": {
        "type": "Qualitativo (max)",
        "definition": "Possibilidade de replicação.",
        "metric": "Likert 1-5",
        "scale": [
            {"Escala": "1", "Descrição": "Não replicável"},
            {"Escala": "2", "Descrição": "Baixa replicabilidade"},
            {"Escala": "3", "Descrição": "Replicabilidade moderada"},
            {"Escala": "4", "Descrição": "Replicável com adaptações"},
            {"Escala": "5", "Descrição": "Altamente replicável"}
        ]
    },
    "Engajamento local": {
        "type": "Qualitativo (max)",
        "definition": "Participação ativa da comunidade.",
        "metric": "Likert 1-5",
        "scale": [
            {"Escala": "1", "Descrição": "Sem engajamento"},
            {"Escala": "2", "Descrição": "Baixo engajamento"},
            {"Escala": "3", "Descrição": "Engajamento moderado"},
            {"Escala": "4", "Descrição": "Alto engajamento"},
            {"Escala": "5", "Descrição": "Engajamento exemplar"}
        ]
    }
}

# =============================
# HEADER SEM IMAGENS (100% ONLINE)
# =============================
def render_header(subtitle: str):
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("**PDM**")
    with col2:
        st.markdown(
            "<h1 style='margin: 0; padding-top: 20px; color: #003087;'>"
            "Processo de Decisão Multicritério para Seleção de Projetos Sociais"
            "</h1>",
            unsafe_allow_html=True
        )
        st.markdown(f"<p style='margin: 0; font-size: 18px; color: #555;'>{subtitle}</p>", unsafe_allow_html=True)
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
    with cols[0]: st.markdown("**Prof. Drª. Luciana Hazin Alencar**")
    with cols[1]: st.markdown("**Gabriel Mendes de Souza**")
    st.markdown("---")
    st.markdown("### Instituições")
    st.markdown("• UFPE – Universidade Federal de Pernambuco")
    st.markdown("• Departamento de Engenharia de Produção")
    st.markdown("• PMD – Gestão e Desenvolvimento de Projetos")
    st.markdown("---")
    st.markdown(
        "**Sobre o sistema**  \n"
        "• PROMETHEE V C-ÓTIMO com **φ* = φ + T**  \n"
        "• Baseado em Vetschera & Almeida (2012)  \n"
        "• 100% funcional no Streamlit Cloud"
    )

def page_promethee_v():
    render_header("PROMETHEE V C-ÓTIMO — Avaliação Multicritério")

    # === SALVAR / CARREGAR ===
    col_save, col_load = st.columns([1, 1])
    with col_save:
        if st.button("**Salvar Sessão (.json)**", use_container_width=True):
            data = save_session_state()
            st.download_button("Baixar Sessão", data=data, file_name="sessao.json", mime="application/json")
    with col_load:
        uploaded = st.file_uploader("**Carregar Sessão**", type="json", label_visibility="collapsed")
        if uploaded:
            load_session_state(uploaded)

    # === PROJETOS E ATORES ===
    st.subheader("Projetos e Atores")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Atores**")
        decision_makers = [s.strip() for s in st.text_input("Decisores", "DM1, DM2").split(',') if s.strip()]
        analysts = [s.strip() for s in st.text_input("Analistas", "Ana, Bruno").split(',') if s.strip()]
        experts = [s.strip() for s in st.text_input("Especialistas", "Carlos, Maria").split(',') if s.strip()]
    with col2:
        st.markdown("**Projetos**")
        if "projects" not in st.session_state:
            st.session_state.projects = ["Projeto A", "Projeto B", "Projeto C"]
        for i, proj in enumerate(st.session_state.projects):
            col_a, col_b = st.columns([5, 1])
            with col_a:
                new_name = st.text_input(f"Projeto {i+1}", proj, key=f"proj_{i}", label_visibility="collapsed")
                st.session_state.projects[i] = new_name.strip() or f"Projeto {i+1}"
            with col_b:
                if st.button("X", key=f"del_{i}"):
                    st.session_state.projects.pop(i)
                    st.rerun()
        if st.button("Adicionar Projeto"):
            st.session_state.projects.append(f"Projeto {len(st.session_state.projects)+1}")
            st.rerun()
        projects = [p for p in st.session_state.projects if p.strip()]
        st.session_state.projects = projects

    # === CRITÉRIOS ===
    st.subheader("1) Critérios")
    all_criteria = list(predefined_criteria.keys())
    selected = st.multiselect("Escolha critérios:", all_criteria, default=all_criteria[:3])
    custom = st.text_input("Personalizados (vírgula)")
    if custom.strip():
        selected += [c.strip() for c in custom.split(',') if c.strip()]
    if not selected:
        st.warning("Selecione pelo menos um critério.")
        return

    if "crit_states" not in st.session_state:
        st.session_state.crit_states = {}
    default_opt = optimize_criteria_default(selected)
    for c in selected:
        if c not in st.session_state.crit_states:
            st.session_state.crit_states[c] = {"direction": default_opt.get(c, "Maximize"), "func": 'u', "q": 0.0, "p": 1.0, "weight": round(1.0/len(selected), 4)}
    for c in list(st.session_state.crit_states.keys()):
        if c not in selected:
            st.session_state.crit_states.pop(c)

    for c in selected:
        st.markdown(f"### {c}")
        if c in predefined_criteria:
            info = predefined_criteria[c]
            st.caption(f"{info['type']} — {info['metric']}")
            with st.expander("Definição"):
                st.write(info['definition'])
            if 'scale' in info:
                with st.expander("Escala Likert"):
                    st.table(pd.DataFrame(info['scale']))
        col_dir, col_func = st.columns(2)
        with col_dir:
            st.session_state.crit_states[c]["direction"] = st.selectbox("Direção", ["Maximize", "Minimize"], 
                index=0 if st.session_state.crit_states[c]["direction"]=="Maximize" else 1, key=f"dir_{c}")
        with col_func:
            func = st.selectbox("Função", ["usual", "v-shape", "linear"], 
                index={"u":0,"v":1,"l":2}[st.session_state.crit_states[c]["func"]], key=f"func_{c}")
            st.session_state.crit_states[c]["func"] = {"usual": 'u', "v-shape": 'v', "linear": 'l'}[func]
        if st.session_state.crit_states[c]["func"] != 'u':
            col_q, col_p = st.columns(2)
            with col_q:
                st.session_state.crit_states[c]["q"] = st.number_input("q", value=st.session_state.crit_states[c]["q"], key=f"q_{c}")
            with col_p:
                st.session_state.crit_states[c]["p"] = st.number_input("p", value=st.session_state.crit_states[c]["p"], key=f"p_{c}")
            if st.session_state.crit_states[c]["p"] <= st.session_state.crit_states[c]["q"]:
                st.error("p deve ser > q")

    # === PESOS ===
    st.subheader("2) Pesos")
    for c in selected:
        st.session_state.crit_states[c]["weight"] = st.slider(f"Peso — {c}", 0.0, 1.0, st.session_state.crit_states[c]["weight"], 0.01, key=f"w_{c}")
    if st.button("Normalizar"):
        total = sum(st.session_state.crit_states[c]["weight"] for c in selected)
        if total > 0:
            for c in selected:
                st.session_state.crit_states[c]["weight"] /= total
            st.success("Normalizado!")

    # === MATRIZ ===
    st.subheader("3) Matriz de Consequência")
    seed = st.number_input("Seed", 0, value=42)
    matrix_df = build_consequence_matrix(projects, selected, seed)
    edit_df = st.data_editor(matrix_df, use_container_width=True, key="matrix")
    cost_criterion = st.selectbox("Critério de CUSTO", selected)
    cost_idx = selected.index(cost_criterion)
    budget = st.number_input("Orçamento", value=float(edit_df.iloc[:, cost_idx].sum()/2))

    # === EXECUTAR ===
    if st.button("**Executar PROMETHEE V C-ÓTIMO**"):
        opt = {c: st.session_state.crit_states[c]["direction"] for c in selected}
        funcs = {c: st.session_state.crit_states[c]["func"] for c in selected}
        th = {c: {"q": st.session_state.crit_states[c]["q"], "p": st.session_state.crit_states[c]["p"]} for c in selected}
        w = [st.session_state.crit_states[c]["weight"] for c in selected]
        if sum(w) == 0:
            st.error("Soma dos pesos = 0")
            return
        matrix = edit_df.to_numpy()
        portfolio, scores, total_cost = promethee_v_c_otimo(matrix, projects, selected, opt, w, funcs, th, budget, cost_idx)

        st.success("**Portfólio Otimizado!**")
        st.write("**Projetos Selecionados:**", ", ".join(portfolio) if portfolio else "Nenhum")
        st.write(f"**Custo Total:** {total_cost:.2f}")

        # FLUXOS
        st.subheader("Fluxos (φ*, T)")
        phi_plus, phi_minus, phi_net, phi_star, T = compute_preference_flows(matrix, selected, opt, w, funcs, th)
        df_fluxos = pd.DataFrame({
            'Projeto': projects,
            'φ⁺': phi_plus.round(4),
            'φ⁻': phi_minus.round(4),
            'φ': phi_net.round(4),
            'φ*': phi_star.round(4)
        }).sort_values('φ*', ascending=False)
        st.dataframe(df_fluxos)

        # GRÁFICO
        fig, ax = plt.subplots()
        x = np.arange(len(projects))
        ax.bar(x-0.3, phi_plus, 0.2, label='φ⁺')
        ax.bar(x-0.1, phi_minus, 0.2, label='φ⁻')
        ax.bar(x+0.1, phi_net, 0.2, label='φ')
        ax.bar(x+0.3, phi_star, 0.2, label='φ*')
        ax.set_xticks(x)
        ax.set_xticklabels(projects, rotation=45)
        ax.legend()
        st.pyplot(fig)

        st.info(f"**T = {T:.4f}** (Vetschera & Almeida, 2012)")

        # EXPORTAR
        st.download_button("Fluxos (CSV)", df_fluxos.to_csv(index=False).encode(), "fluxos.csv")
        st.download_button("Portfólio (CSV)", pd.DataFrame({'Projeto': portfolio}).to_csv(index=False).encode(), "portfolio.csv")

# =============================
# MENU
# =============================
menu = ["Home", "PROMETHEE V-C-ÓTIMO"]
choice = st.sidebar.selectbox("Menu", menu)
if choice == "Home":
    page_home()
else:
    page_promethee_v()

st.caption("© PDMSPS — UFPE | PMD | 2025")
