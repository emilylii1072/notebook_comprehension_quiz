"""Fixed content for the pre- and post-study surveys.

Transcribed directly from the study's two Qualtrics survey exports
("Delegations_Pre-Survey.qsf" / "Delegations_Post-Survey.qsf") -- no LLM
generation, no admin editing. Each .qsf's Survey Flow names which blocks are
actually administered; a "Trash / Unused Questions" block in both files holds
retired questions (an old code-snippet item, a duplicate p-value item, "What
parts did the AI do for this task?", ...) that are no longer part of either
survey and are not transcribed here. The small italicized prefixes on each
question ("Logistics", "AI trust", "Machine Learning 2", etc. -- Qualtrics'
"Data Export Tag") are kept as `category` for analysis, not part of the
question text; where the export sets no tag, `category` is None.

One content decision: the post-survey's "Confidence" matrix (QID7) repeats "I
understand why the analysis produced its conclusions." as both its 1st and
4th row -- a pagination artifact from an earlier print layout -- so it's
asked once here, not twice. Its 5th/6th rows also read "...for the same task
task with/without AI assistance" (a doubled word) in the .qsf; transcribed
here as "task" once. All four post-survey workload ("Experience") blocks
share the same 5 sub-questions, though the 3rd ("How irritated...") is worded
per-task ("task" vs "tasks") exactly as each block has it.

A question with `force=OFF` in its .qsf Validation settings is optional here
(`SurveyItem.optional=True`); everything else is required. Subject ID is
never re-asked here -- the participant already entered it at the "identify"
stage of app_pages/participant.py.
"""

from dataclasses import dataclass


@dataclass
class SurveyItem:
    id: str
    kind: str  # "notice" | "short_text" | "single_select" | "multi_select" | "matrix" | "long_text"
    question: str
    category: str | None = None
    options: list[str] | None = None      # single/multi_select choices, or a matrix's column scale
    rows: list[str] | None = None         # matrix row statements
    free_text_options: tuple[str, ...] = ()  # choices that reveal a text box when chosen, e.g. ("Other",)
    optional: bool = False                # force=OFF in the .qsf -- skippable, not required to continue
    min_value: int | None = None          # short_text only -- a ValidNumber range from the .qsf
    max_value: int | None = None          # (both set together); rendered as a number input, not free text


_KNOWLEDGE_NOTICE = (
    "Please answer these questions to the best of your ability. It is "
    "completely okay to not know the answer to these questions. If you do "
    "not know an answer to the questions, you may write \"I don't know\" or "
    "\"idk\". Please do not consult any external sources!"
)

_AGREE_5 = ["Strongly disagree", "Disagree", "Neither agree nor disagree", "Agree", "Strongly agree"]
_AGREE_5_REVERSED = ["Strongly agree", "Agree", "Neither agree nor disagree", "Disagree", "Strongly disagree"]
_FREQ_5 = ["Never", "Rarely", "Monthly", "Weekly", "Daily"]
_EXTENT_5 = ["Not at all", "Slightly", "Moderately", "Very", "Extremely"]
_AI_ROLE_OPTIONS = [
    "Autonomous — AI agent handles the task entirely on its own",
    "Minimal Input — AI agent requires minimal human input for optimal performance",
    "Partnership — AI agent and human form an equal partnership, outperforming either alone",
    "Human-Led — AI agent requires human input to successfully complete the task",
    "Full Involvement — AI agent cannot function without continuous human involvement",
]


def _workload_block(item_id: str, title: str, irritated_wording: str) -> SurveyItem:
    return SurveyItem(
        id=item_id, kind="matrix", category=None,
        question=f"Please indicate the extent to which each question describes your experience regarding {title}:",
        rows=[
            "How mentally demanding or difficult was this task?",
            "How hard/effortful did you have to work to complete this task?",
            f"How irritated, stressed, or frustrated did you feel during the {irritated_wording}?",
            "How hurried or rushed was the pace of the task?",
            "How familiar are you with the type of task?",
        ],
        options=_EXTENT_5,
    )


PRE_SURVEY_ITEMS: list[SurveyItem] = [
    SurveyItem(
        id="logistics_age", kind="short_text", category="Logistics",
        question="How old are you?",
        min_value=18, max_value=99,
    ),
    SurveyItem(
        id="logistics_gender", kind="single_select", category="Logistics",
        question="What is your gender?",
        options=["Male", "Female", "Other (please specify)", "Prefer not to say"],
        free_text_options=("Other (please specify)",),
    ),
    SurveyItem(
        id="logistics_first_language", kind="multi_select", category="Logistics",
        question="What is your first language?",
        options=[
            "English", "Chinese (Mandarin)", "Chinese (Cantonese)", "Spanish", "Japanese",
            "Hindi", "Korean", "Arabic", "German", "Italian", "Other",
        ],
        free_text_options=("Other",),
    ),
    SurveyItem(
        id="logistics_occupation", kind="multi_select", category="Logistics",
        question=(
            "What best describes your current occupation? If you are a student, "
            "indicate your major."
        ),
        options=[
            "Data Scientist / Analyst", "Software Engineering", "Product Managing",
            "Designer / Researcher", "Student", "Other",
        ],
        free_text_options=("Student", "Other"),
    ),
    SurveyItem(
        id="prior_experience_ds_level", kind="single_select", category="Prior experience",
        question="How experienced are you with data science?",
        options=[
            "Not at all experienced", "Slightly experienced", "Moderately experienced",
            "Very experienced", "Extremely experienced",
        ],
    ),
    SurveyItem(
        id="prior_experience_skill_freq", kind="matrix", category="Prior experience",
        question="How frequently do you use each of these skills in your work?",
        rows=["Machine Learning", "Data Visualization", "Data Analysis"],
        options=_FREQ_5,
    ),
    SurveyItem(
        id="ai_trust_statements", kind="matrix", category="AI trust",
        question="Please rate the following in terms of how much you agree or disagree with each statement.",
        rows=[
            "I am confident in my ability to use GenAI/LLMs.",
            "I am comfortable letting an AI tool suggest what to do next in a task.",
            "Generally, I trust AI-created outputs.",
        ],
        options=_AGREE_5,
    ),
    SurveyItem(
        id="ai_experience_freq", kind="matrix", category="AI experience",
        question="How frequently do you use GenAI/LLM (like ChatGPT, Claude, etc) for:",
        rows=["In general", "Programming", "Data Analysis"],
        options=_FREQ_5,
    ),
    SurveyItem(
        id="ai_experience_interactions", kind="multi_select", category="AI experience",
        question="Select all of the ways you have interacted with AI",
        options=[
            "Conversed with LLM tools such that ChatGPT, Claude, Gemini (e.g. ask questions)",
            "Used agentic AI for projects",
            "Written reusable skills like SKILL.MD for agents",
            "Building the MCP servers, APIs, or environments it operates in",
            "None",
            "Other",
        ],
        free_text_options=("Other",),
    ),
    SurveyItem(
        id="ai_collab_role", kind="single_select", category="AI collab",
        question="Reflecting on your typical work tasks, which best describes the role AI agents can play in your job?",
        options=_AI_ROLE_OPTIONS,
    ),
    SurveyItem(
        id="ai_collab_time_pref", kind="single_select", category="AI collab",
        question=(
            "When working on a data science task, how would you like to spend the "
            "majority of your time interacting with AI?"
        ),
        options=["Plan myself + AI implements", "AI implements + I evaluate", "Other"],
        free_text_options=("Other",),
    ),
    SurveyItem(
        id="ai_collab_learn_strategies", kind="matrix", category="AI collab",
        question="How likely are you to use the following strategies to learn something new?",
        rows=[
            "Let AI summarize/break down the topic for me",
            "Ask AI questions for AI to answer",
            "Iterate your thoughts/answers with AI and ask AI to critique",
            "Let AI ask you questions and critique your answer",
        ],
        options=[
            "Extremely unlikely", "Unlikely", "Slightly Unlikely", "Neutral",
            "Slightly likely", "Likely", "Extremely likely",
        ],
        optional=True,
    ),
    SurveyItem(id="pre_knowledge_notice", kind="notice", question=_KNOWLEDGE_NOTICE),
    SurveyItem(
        id="know_ds1", kind="single_select", category="Data science",
        question="Data science 1. What is the primary goal of exploratory data analysis (EDA)?",
        options=[
            "To inspect the data for structure, issues, and potentially useful patterns",
            "To determine whether the null hypothesis should be rejected as early as possible",
            "To reduce the dataset to only the statistically significant variables",
            "To ensure all variables are normally distributed before analysis begins",
            "I don't know",
        ],
    ),
    SurveyItem(
        id="know_prog1", kind="single_select", category="Programming",
        question=(
            "Programming 1. Suppose the dataset has 1000 rows of student scores and "
            "200 students. After the following code, how many rows will person_df "
            'have? person_df = df.groupby("PersonId").mean(numeric_only=True).reset_index()'
        ),
        options=["1000", "200", "800", "It must be equal to the number of columns", "I don't know"],
    ),
    SurveyItem(
        id="know_ml1", kind="single_select", category="Machine Learning",
        question="Machine Learning 1. What is a confusion matrix used for in machine learning?",
        options=[
            "To visualize the correlation between numerical features",
            "To summarize comparison between predicted vs. actual labels",
            "To detect missing values in a dataset",
            "To scale features before training a model",
            "I don't know",
        ],
    ),
    SurveyItem(
        id="know_ml2", kind="single_select", category="Machine Learning",
        question=(
            "Machine Learning 2. Which model does this describe: This model builds "
            "decision trees sequentially, where each new tree focuses on correcting "
            "the errors of the previous trees by fitting to their residuals. It uses "
            "gradient descent to minimize a loss function and includes built-in "
            "regularization to help prevent overfitting."
        ),
        options=["Random Forest", "K-Means", "Logistic Regression", "XG-Boost", "I don't know"],
    ),
    SurveyItem(
        id="know_ml3", kind="single_select", category="Machine Learning",
        question=(
            "Machine Learning 3. Which model does this describe: a supervised model "
            "used to estimate the probability that an employee will leave the "
            "company, where the outcome has two classes and each feature contributes "
            "through an interpretable coefficient?"
        ),
        options=["Random Forest", "K-Means", "Logistic Regression", "XG-Boost", "I don't know"],
    ),
    SurveyItem(
        id="know_ml4", kind="single_select", category="Machine Learning",
        question=(
            "Machine Learning 4. A model trains many decision trees on different "
            "samples of the data and combines their predictions by voting. Which "
            "model is being used?"
        ),
        options=["Random Forest", "K-Means", "K-Nearest Neighbors", "XG-Boost", "I don't know"],
    ),
    SurveyItem(
        id="know_ml5", kind="single_select", category="Machine Learning",
        question='Machine Learning 5. What is "feature engineering" in the context of machine learning?',
        options=[
            "The process of writing production code to deploy a trained model",
            "The process of creating, transforming, or selecting input variables from raw data",
            "The process of selecting which pre-existing columns in a dataset to keep and which to discard before training",
            "The process of adjusting a model's internal parameters (like tree depth or learning rate) to improve performance on validation data",
            "I don't know",
        ],
    ),
    SurveyItem(
        id="know_ds2", kind="single_select", category="Data science",
        question="Data Science 2. What does aggregation typically refer to?",
        options=[
            "Removing duplicate rows from a dataset before analysis",
            "Combining multiple raw records into a single summary value or row",
            "Merging two separate datasets together based on a shared key column",
            "Converting a categorical variable into multiple binary indicator columns",
            "I don't know",
        ],
    ),
    SurveyItem(
        id="know_ds3", kind="single_select", category="Data science",
        question=(
            "Data Science 3. What is this evaluation metric: This metric plots the "
            "true positive rate against the false positive rate at every possible "
            "decision threshold, then summarizes performance as the area under that curve."
        ),
        options=["Precision-Recall Curve", "F1 Score", "ROC-AUC", "Confusion Matrix", "I don't know"],
    ),
    SurveyItem(
        id="know_ml6", kind="single_select", category="Machine Learning",
        question=(
            "Machine Learning 6. Is the steps correct: Fills missing values using the "
            "median calculated from the entire dataset. Standardizes all features "
            "using the mean and standard deviation from the entire dataset. Splits "
            "the processed data into an 100% training set and a 20% test set. Trains "
            "a Logistic Regression model on the training set. Evaluates it on the "
            "test set and reports."
        ),
        options=["Yes", "No", "I don't know"],
        optional=True,
    ),
]


POST_SURVEY_ITEMS: list[SurveyItem] = [
    SurveyItem(id="post_knowledge_notice", kind="notice", question=_KNOWLEDGE_NOTICE),
    SurveyItem(
        id="post_know_eda", kind="single_select", category="Data science",
        question="What is the primary goal of exploratory data analysis (EDA)?",
        options=[
            "To inspect the data for structure, issues, and potentially useful patterns",
            "To determine whether the null hypothesis should be rejected as early as possible",
            "To reduce the dataset to only the statistically significant variables",
            "To ensure all variables are normally distributed before analysis begins",
            "I don't know",
        ],
    ),
    SurveyItem(
        id="post_know_groupby", kind="single_select", category="Programming",
        question=(
            "Suppose the original dataset has 5,000 rows and 300 employees. After "
            "the following code, how many rows will person_df have? person_df = "
            'df.groupby("PersonId").mean(numeric_only=True).reset_index()'
        ),
        options=["5000", "300", "4700", "It must be equal to the number of columns", "I don't know"],
    ),
    SurveyItem(
        id="post_know_confusion", kind="single_select", category="Machine Learning",
        question="What is a confusion matrix used for in machine learning?",
        options=[
            "To visualize the correlation between numerical features",
            "To summarize comparison between predicted vs. actual labels",
            "To detect missing values in a dataset",
            "To scale features before training a model",
            "I don't know",
        ],
    ),
    SurveyItem(
        id="post_know_gboost", kind="single_select", category="Machine Learning",
        question=(
            "Which model does this describe: This model builds decision trees "
            "sequentially, where each new tree focuses on correcting the errors of "
            "the previous trees by fitting to their residuals. It uses gradient "
            "descent to minimize a loss function and includes built-in "
            "regularization to help prevent overfitting."
        ),
        options=["Random Forest", "K-Means", "Logistic Regression", "XG-Boost", "I don't know"],
    ),
    SurveyItem(
        id="post_know_logreg", kind="single_select", category="Machine Learning",
        question=(
            "Which model does this describe: a supervised model used to estimate "
            "the probability that an employee will leave the company, where the "
            "outcome has two classes and each feature contributes through an "
            "interpretable coefficient?"
        ),
        options=["Random Forest", "K-Means", "Logistic Regression", "XG-Boost", "I don't know"],
    ),
    SurveyItem(
        id="post_know_rf", kind="single_select", category="Machine Learning",
        question=(
            "What model does this describe: This model builds many decision trees "
            "independently in parallel, each trained on a random subset of "
            "employees and features, then averages their predictions to classify "
            "attrition risk."
        ),
        options=["Random Forest", "K-Means", "K-Nearest Neighbors", "XG-Boost", "I don't know"],
    ),
    SurveyItem(
        id="post_know_feature_eng", kind="single_select", category="Machine Learning",
        question='What is "feature engineering" in the context of machine learning?',
        options=[
            "The process of writing production code to deploy a trained model",
            "The process of creating, transforming, or selecting input variables from raw data",
            "The process of selecting which pre-existing columns in a dataset to keep and which to discard before training",
            "The process of adjusting a model's internal parameters (like tree depth or learning rate) to improve performance on validation data",
            "I don't know",
        ],
    ),
    SurveyItem(
        id="post_know_aggregation", kind="single_select", category="Data science",
        question="What does aggregation typically refer to?",
        options=[
            "Removing duplicate rows from a dataset before analysis",
            "Combining multiple raw records into a single summary value or row",
            "Merging two separate datasets together based on a shared key column",
            "Converting a categorical variable into multiple binary indicator columns",
            "I don't know",
        ],
    ),
    SurveyItem(
        id="post_know_rocauc", kind="single_select", category="Data science",
        question=(
            "What is this evaluation metric: This metric plots the true positive "
            "rate against the false positive rate at every possible decision "
            "threshold, then summarizes performance as the area under that curve."
        ),
        options=["Precision-Recall Curve", "F1 Score", "ROC-AUC", "Confusion Matrix", "I don't know"],
    ),
    SurveyItem(
        id="post_know_pipeline", kind="single_select", category="Machine Learning",
        question=(
            "Is the steps correct: Fills missing values using the median "
            "calculated from the entire dataset. Standardizes all features using "
            "the mean and standard deviation from the entire dataset. Splits the "
            "processed data into an 100% training set and a 20% test set. Trains "
            "a Logistic Regression model on the training set. Evaluates it on the "
            "test set and reports."
        ),
        options=["Yes", "No", "I don't know"],
    ),
    SurveyItem(
        id="post_confidence_matrix", kind="matrix", category="AI trust",
        question="Please rate the following in terms of how much you agree or disagree with each statement.",
        rows=[
            "I am confident in my ability to use GenAI/LLMs.",
            "I am comfortable letting an AI tool suggest what to do next in a task.",
            "I trusted AI-created output.",
        ],
        options=_AGREE_5,
    ),
    SurveyItem(
        id="post_reflection_matrix", kind="matrix", category="Confidence",
        question="How much do you agree/disagree with the following statements:",
        rows=[
            "I understand why the analysis produced its conclusions.",
            "I could explain the analysis to another person without relying on the AI.",
            "I actively checked whether the AI's outputs were correct.",
            "I am comfortable code reviewing other people's Python notebooks for the same task with AI assistance",
            "I am comfortable code reviewing other people's Python notebooks for the same task without AI assistance",
            "I am comfortable generating new analysis ideas based on what I learned from this task.",
        ],
        options=_AGREE_5_REVERSED,
    ),
    SurveyItem(
        id="post_ai_role", kind="single_select",
        question="Reflecting on your experience with the tasks today, which best describes the role of AI agents?",
        options=_AI_ROLE_OPTIONS,
        optional=True,
    ),
    _workload_block("workload_main_task", "the main task (Python notebook)", "task"),
    _workload_block("workload_idea_gen", "the idea generation task", "tasks"),
    _workload_block("workload_debugging", "the debugging task", "tasks"),
    _workload_block(
        "workload_interview",
        "the interview part of the study (verbally answering questions)", "task",
    ),
    SurveyItem(
        id="post_grade_validity", kind="short_text",
        question="If you were to grade your notebook, what grade would you give for its validity? (0-4)",
    ),
    SurveyItem(
        id="post_grade_completeness", kind="short_text",
        question="If you were to grade your notebook, what grade would you give for its completeness? (0-4)",
    ),
    SurveyItem(
        id="post_ai_collab_strategy", kind="long_text",
        question=(
            "What AI-collaboration strategy do you think would help you further "
            "improve your performance when working with AI on these kinds of data "
            "analysis activities?"
        ),
    ),
    SurveyItem(
        id="post_demanding_part", kind="long_text",
        question="Did you find any part of the task particularly demanding / challenging?",
    ),
    SurveyItem(
        id="post_other_comments", kind="long_text", question="Anything else you want to share?",
        optional=True,
    ),
]


# ---------------------------------------------------------------------------
# Scoring the knowledge assessments
# ---------------------------------------------------------------------------
#
# The Qualtrics exports record the questions but not the answer key, so the
# key below was written out here rather than transcribed. Every entry is a standard
# data-science fact with one defensible answer (the ML6 pipeline item is "No"
# because it both leaks the test set through whole-dataset imputation/scaling
# and splits into 100%/20%), but it is a judgement call rather than a source
# document -- check it before reporting any pre/post learning gain.
#
# Keys are SurveyItem.id; values are the exact option string that counts as
# correct. Items not listed here are opinion/experience questions and are never
# scored.

ANSWER_KEY: dict[str, str] = {
    # --- pre-survey ---
    "know_ds1": "To inspect the data for structure, issues, and potentially useful patterns",
    "know_prog1": "200",
    "know_ml1": "To summarize comparison between predicted vs. actual labels",
    "know_ml2": "XG-Boost",
    "know_ml3": "Logistic Regression",
    "know_ml4": "Random Forest",
    "know_ml5": "The process of creating, transforming, or selecting input variables from raw data",
    "know_ds2": "Combining multiple raw records into a single summary value or row",
    "know_ds3": "ROC-AUC",
    "know_ml6": "No",
    # --- post-survey ---
    "post_know_eda": "To inspect the data for structure, issues, and potentially useful patterns",
    "post_know_groupby": "300",
    "post_know_confusion": "To summarize comparison between predicted vs. actual labels",
    "post_know_gboost": "XG-Boost",
    "post_know_logreg": "Logistic Regression",
    "post_know_rf": "Random Forest",
    "post_know_feature_eng": "The process of creating, transforming, or selecting input variables from raw data",
    "post_know_aggregation": "Combining multiple raw records into a single summary value or row",
    "post_know_rocauc": "ROC-AUC",
    "post_know_pipeline": "No",
}

# (pre item id, post item id, short label) -- the same concept asked before and
# after, so a participant's pre/post pair is directly comparable. The two
# groupby items deliberately use different numbers so the second isn't a memory
# test; both are scored against their own row in ANSWER_KEY.
KNOWLEDGE_PAIRS: list[tuple[str, str, str]] = [
    ("know_ds1", "post_know_eda", "EDA goal"),
    ("know_prog1", "post_know_groupby", "groupby row count"),
    ("know_ml1", "post_know_confusion", "Confusion matrix"),
    ("know_ml2", "post_know_gboost", "Gradient boosting"),
    ("know_ml3", "post_know_logreg", "Logistic regression"),
    ("know_ml4", "post_know_rf", "Random forest"),
    ("know_ml5", "post_know_feature_eng", "Feature engineering"),
    ("know_ds2", "post_know_aggregation", "Aggregation"),
    ("know_ds3", "post_know_rocauc", "ROC-AUC"),
    ("know_ml6", "post_know_pipeline", "Leakage in pipeline"),
]

# Statements asked on both surveys, so the shift is measurable. The trust row is
# reworded between the two ("Generally, I trust AI-created outputs." ->
# "I trusted AI-created output."), hence pre and post row text both appear.
MATCHED_LIKERT: list[tuple[str, str, str, str, str]] = [
    ("ai_trust_statements", "I am confident in my ability to use GenAI/LLMs.",
     "post_confidence_matrix", "I am confident in my ability to use GenAI/LLMs.",
     "Confidence using GenAI/LLMs"),
    ("ai_trust_statements", "I am comfortable letting an AI tool suggest what to do next in a task.",
     "post_confidence_matrix", "I am comfortable letting an AI tool suggest what to do next in a task.",
     "Comfort letting AI suggest next step"),
    ("ai_trust_statements", "Generally, I trust AI-created outputs.",
     "post_confidence_matrix", "I trusted AI-created output.",
     "Trust in AI output"),
]

# Likert scales, most-negative first, for ordering a stacked distribution. A
# matrix item's own `options` may be reversed (the post-survey shows some scales
# strongly-agree-first); charts order by these, not by presentation order.
LIKERT_SCALES: list[list[str]] = [_AGREE_5, _FREQ_5, _EXTENT_5, [
    "Extremely unlikely", "Unlikely", "Slightly Unlikely", "Neutral",
    "Slightly likely", "Likely", "Extremely likely",
]]

ITEMS_BY_ID: dict[str, SurveyItem] = {
    it.id: it for it in (*PRE_SURVEY_ITEMS, *POST_SURVEY_ITEMS)
}


def selected_option(answer) -> str | None:
    """The chosen option of a single_select answer, however it was stored."""
    if isinstance(answer, dict):
        sel = answer.get("selected")
        return sel if isinstance(sel, str) else None
    return answer if isinstance(answer, str) else None


def is_correct(item_id: str, answer) -> bool | None:
    """True/False for a scored knowledge item, None if the item isn't scored or
    was left blank."""
    key = ANSWER_KEY.get(item_id)
    if key is None:
        return None
    chosen = selected_option(answer)
    return None if chosen is None else chosen == key


def scale_for(item: SurveyItem) -> list[str] | None:
    """The canonical (negative-first) ordering for a matrix item's scale."""
    opts = set(item.options or [])
    for scale in LIKERT_SCALES:
        if set(scale) == opts:
            return scale
    return None
