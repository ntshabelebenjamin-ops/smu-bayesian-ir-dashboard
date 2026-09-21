import streamlit as st
import pandas as pd, numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="SMU FTEN Bayesian IR Lab",page_icon="📊",layout="wide")
st.title("SMU FTEN Bayesian Institutional Research Lab")
st.caption("Student success • equity • uncertainty • institutional sustainability")
st.success("Privacy-protected deployment: direct student identifiers have been removed from the hosted analytical datasets. The dashboard presents cohort-level and model-level evidence.")

@st.cache_data
def load_data():
    cohort=pd.read_csv("SMU_649_CLEAN_COHORT.csv")
    train=pd.read_csv("SMU_246_TRAIN_OBSERVED_OUTCOME.csv")
    core=pd.read_csv("SMU_246_CORE_MODEL_READY.csv")
    return cohort,train,core
cohort,train,core=load_data()

with st.sidebar:
    st.header("Model controls")
    prior_sd=st.slider("Coefficient prior SD",0.25,2.0,1.0,0.05)
    success_threshold=st.slider("Display success threshold",0.50,1.0,0.75,0.01)
    st.caption("The packaged training target is ≥75% of HEMIS credits passed. Change the threshold only for descriptive outcome views unless you rebuild training data.")
    st.warning("Decision-support only. Do not use this model for automated exclusion, admission, funding or disciplinary decisions.")

tabs=st.tabs(["Executive story","Data pipeline","Cohort profile","Variables & theory","Bayesian model","Validation","Student-support lens","Implications & future research"])

with tabs[0]:
    st.subheader("Financial Sustainability of Universities: the Strategic Role of Institutional Research")
    st.markdown("""
Universities must expand access while protecting quality, student success and institutional sustainability.
For Institutional Research, the strategic question is not only **how many students entered**, but:
**who progresses, under what conditions, with what uncertainty, and where can institutional support make a difference?**

For an HDI context, pre-entry circumstances should not be treated as student deficits. They describe unequal
schooling, household, financial and transition conditions. The IR task is to convert those conditions into
evidence for support, planning and resource allocation.
""")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("FTEN cohort",f"{len(cohort):,}")
    c2.metric("Outcome observed",f"{train.shape[0]:,}")
    c3.metric("≥75% credits",f"{int((train.target_success_75==1).sum())} ({train.target_success_75.mean():.1%})")
    c4.metric("<75% credits",f"{int((train.target_success_75==0).sum())} ({1-train.target_success_75.mean():.1%})")
    st.info("The 403 FTEN students without matched academic outcomes remain part of the cohort profile but are not assigned invented outcomes and are not used as labelled training cases.")

with tabs[1]:
    st.subheader("Step-by-step preprocessing and cleaning")
    steps=pd.DataFrame([
      [1,"Define cohort","Start from FTEN list; validate student number; remove invalid/duplicate IDs.","649 unique valid FTEN students"],
      [2,"Questionnaire","Keep latest valid questionnaire response per student.","Pre-entry/context variables"],
      [3,"Academic outcomes","Match HEMIS credits by student number.","246 observed outcomes"],
      [4,"Engineer outcome","Credit pass rate = HEMIS credits passed / HEMIS credits.","Continuous outcome"],
      [5,"Define target","Success = credit pass rate ≥ 75%.","229 success; 17 below threshold"],
      [6,"Engineer constructs","School facilities, home resources, devices and concern domains.","Transparent derived variables"],
      [7,"Respect skip logic","No dependants → 0 dependants; no part-time work → 0 hours.","Avoid artificial missingness"],
      [8,"Protect outcome","Never impute HEMIS outcomes.","No fabricated labels"],
      [9,"Prevent leakage","Do not use credits passed/failed to predict the target they define.","Valid predictor matrix"],
      [10,"Model preparation","Ordinal encode selected variables; standardise counts; impute predictor gaps only.","Core training matrix"],
    ],columns=["Step","Stage","What happens","Output"])
    st.dataframe(steps,use_container_width=True,hide_index=True)
    st.download_button("Download 649-student clean cohort",cohort.to_csv(index=False),"SMU_649_CLEAN_COHORT.csv","text/csv")
    st.download_button("Download 246 outcome-observed training data",train.to_csv(index=False),"SMU_246_TRAIN_OBSERVED_OUTCOME.csv","text/csv")
    st.download_button("Download final core model-ready data",core.to_csv(index=False),"SMU_246_CORE_MODEL_READY.csv","text/csv")

with tabs[2]:
    st.subheader("Cohort profile")
    var=st.selectbox("Explore a variable",[
      "school_location","home_location","english_language_level","family_attended_university",
      "programme_choice","nsfas_approved","matric_class_size","travel_time_to_8am",
      "school_facilities_count","home_resources_count","concern_any"
    ])
    tmp=cohort[var].fillna("Missing").astype(str).value_counts().reset_index()
    tmp.columns=[var,"Students"]
    st.plotly_chart(px.bar(tmp,x=var,y="Students",title=f"FTEN cohort: {var.replace('_',' ').title()}"),use_container_width=True)
    st.dataframe(tmp,use_container_width=True,hide_index=True)
    st.subheader("Psychosocial concern indicators")
    cc=["concern_academic","concern_financial","concern_adjustment_belonging","concern_accommodation_transport","concern_safety_health","concern_time_management"]
    vals=pd.DataFrame({"Concern":[x.replace("concern_","").replace("_"," ").title() for x in cc],
                       "Students":[int(cohort[x].sum()) for x in cc]})
    st.plotly_chart(px.bar(vals,x="Concern",y="Students"),use_container_width=True)
    st.caption("Concern categories are transparent keyword-based indicators from the open-text first-year fears/concerns item. They are descriptive and may overlap.")

with tabs[3]:
    st.subheader("Why these variables?")
    framework=pd.DataFrame([
      ["Schooling context","School location, class size, facilities, online learning","Prior opportunity-to-learn and resource environment","Available"],
      ["Language transition","Teaching language, English level, home/school language alignment","Potential transition into university academic discourse","Available"],
      ["Household resources","Water/electricity/computer/internet/study space/backup power","Material and digital learning capital","Available"],
      ["First-generation context","Family attended university","Proxy for familiarity with university systems/social capital","Available"],
      ["Financial support","NSFAS application/approval and funding source","Financial access and continuity","Available"],
      ["Academic fit","Programme choice","Alignment between intended and enrolled programme","Available"],
      ["Time/access","Travel, distance, part-time work, dependants","Potential time poverty and competing responsibilities","Available"],
      ["Psychosocial transition","First-year fears/concerns","Adjustment pressures and support needs","Partially available"],
      ["Matric achievement/quintile","APS, subject marks, school quintile","Important future comparator for CPUT model","Not in SMU files"],
    ],columns=["Domain","SMU variables","IR interpretation","Status"])
    st.dataframe(framework,use_container_width=True,hide_index=True)
    st.markdown("""
**Policy/literature framing used in this study**

- DHET's 2025–2030 strategy and 2026/27 planning emphasise an equitable, efficient and sustainable PSET system and enrolment planning within resource constraints.
- CHE work on historically disadvantaged students emphasises the distinction between formal access and epistemic access/success.
- CHE's 2026 Higher Education Monitor 18 explicitly places **funding, access and success** in the same analytical frame.
- Universities South Africa identifies long-term financial sustainability, student debt and NSFAS sustainability among current sector priorities.

The model therefore treats student success as both an **equity question** and a **strategic sustainability question**.
""")

def fit_bayes(data,prior_sd=1.0):
    feats=[c for c in data.columns if c != "target_success_75"]
    X0=data[feats].astype(float).values
    y=data.target_success_75.astype(int).values
    X=np.column_stack([np.ones(len(y)),X0])
    base=np.clip(y.mean(),1e-6,1-1e-6)
    mu=np.r_[np.log(base/(1-base)),np.zeros(len(feats))]
    sd=np.r_[1.5,np.repeat(prior_sd,len(feats))]
    prec=1/sd**2
    def obj(t): return np.sum(np.logaddexp(0,X@t)-y*(X@t))+.5*np.sum((t-mu)**2*prec)
    def grad(t): return X.T@(expit(X@t)-y)+(t-mu)*prec
    fit=minimize(obj,mu,jac=grad,method="BFGS")
    t=fit.x; p=expit(X@t); w=p*(1-p)
    cov=np.linalg.inv(X.T@(X*w[:,None])+np.diag(prec)); se=np.sqrt(np.diag(cov))
    rows=[]
    for i,n in enumerate(["Intercept"]+feats):
        lo=t[i]-1.96*se[i]; hi=t[i]+1.96*se[i]
        rows.append([n,t[i],np.exp(t[i]),np.exp(lo),np.exp(hi),norm.cdf(t[i]/se[i])])
    return pd.DataFrame(rows,columns=["Variable","Beta","Odds ratio","CrI lower","CrI upper","P(beta>0)"]),p,y

with tabs[4]:
    st.subheader("Bayesian explanatory model")
    res,p,y=fit_bayes(core,prior_sd)
    show=res[res.Variable!="Intercept"].copy()
    st.dataframe(show.style.format({"Beta":"{:.3f}","Odds ratio":"{:.2f}","CrI lower":"{:.2f}","CrI upper":"{:.2f}","P(beta>0)":"{:.1%}"}),use_container_width=True)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=show["Odds ratio"],y=show["Variable"],mode="markers",
        error_x=dict(type="data",symmetric=False,array=show["CrI upper"]-show["Odds ratio"],arrayminus=show["Odds ratio"]-show["CrI lower"])))
    fig.add_vline(x=1,line_dash="dash")
    fig.update_layout(title="Posterior odds ratios with 95% credible intervals",xaxis_title="Odds ratio",yaxis_title="")
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("""
**Lay interpretation:** an odds ratio above 1 points toward higher estimated odds of reaching the credit threshold;
below 1 points toward lower estimated odds. The credible interval shows uncertainty. If it spans 1, the current
data do not provide strong directional certainty.

**Important:** association is not causation. A student's background is not destiny.
""")

with tabs[5]:
    st.subheader("Does the model generalise?")
    # One reproducible 5-fold CV for interactive speed
    feats=[c for c in core.columns if c != "target_success_75"]
    X=core[feats].values; y=core.target_success_75.astype(int).values
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=2026)
    oof=np.zeros(len(y))
    for tr,te in skf.split(X,y):
        trdf=core.iloc[tr].copy()
        r,_,_=fit_bayes(trdf,prior_sd)
        beta=r.Beta.values
        oof[te]=expit(np.column_stack([np.ones(len(te)),X[te]])@beta)
    auc=roc_auc_score(y,oof); brier=brier_score_loss(y,oof)
    c1,c2,c3=st.columns(3)
    c1.metric("5-fold CV AUC",f"{auc:.3f}")
    c2.metric("Brier score",f"{brier:.3f}")
    c3.metric("Below-75% cases",f"{int((y==0).sum())}")
    st.warning("Because only 17 outcome-observed students are below the 75% threshold, predictive discrimination is unstable. Treat this as an explanatory/exploratory model, not a production risk engine.")
    st.markdown("A model can look useful in the training sample and still fail on unseen students. Cross-validation is therefore displayed alongside the posterior effects.")

with tabs[6]:
    st.subheader("Student-support lens — not a deficit label")
    st.markdown("""
The dashboard should support conversations such as:

- **Resource constraint:** Does a student lack internet, a study space, reliable power or learning devices?
- **Transition:** Is university the student's first exposure in their close family?
- **Financial continuity:** Is NSFAS/funding unresolved?
- **Academic fit:** Is the student studying their first-choice programme?
- **Time poverty:** Is commuting, employment or caring responsibility likely to reduce study time?
- **Psychosocial transition:** What concerns did the student themselves identify?

The intervention logic is **support-first**. These variables should not be used to deny admission, funding or opportunity.
""")
    st.info("For a layperson: the model says 'students with these circumstances showed this pattern in this cohort, with this much uncertainty' — not 'this student will fail'.")

with tabs[7]:
    st.subheader("Executive and sector implications")
    st.markdown("""
### For SMU executives
Use the evidence to connect **access → transition support → progression → retention → completion**. Weak predictive
performance is itself strategically useful: it warns against simplistic profiling based on socioeconomic background.

### For financial sustainability
Attrition and delayed progression affect students and consume scarce teaching, support and infrastructure capacity.
IR can help test whether targeted support is reaching the conditions associated with progression, rather than merely
reporting historical success rates.

### For HDIs
The analysis should resist a deficit narrative. Structural disadvantage is context for institutional response, not
evidence of lower ability.

### Future comparative research: SMU + CPUT
The SMU model uses rich biographical/contextual data. The CPUT model will use a different predictor architecture,
including matric results. The institutions should first be modelled separately. Future research can then ask which
effects generalise across contexts using a multi-institutional or hierarchical Bayesian model.
""")
    st.subheader("Next research design")
    st.code("SMU contextual Bayesian model  →  CPUT matric/academic Bayesian model  →  cross-institutional comparison  →  hierarchical Bayesian model",language=None)
