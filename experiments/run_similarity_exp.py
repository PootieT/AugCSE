from evaluate import parser
from TestRunner import get_implementation
from evaluation.evaluation_engine import evaluate

args = parser.parse_args()
args.transformation = "YodaPerturbation"
args.language = "en"
args.task_type = "SIMILARITY_EXP"
args.model = "sentence-transformers/all-mpnet-base-v2"
args.dataset = "imdb"
# args.percentage_of_examples = "1"  # 20
args.batch_size = 128

# Identify the transformation that the user has mentioned.
if_filter = args.transformation is None
if args.transformation:
    implementation = get_implementation(args.transformation)
else:
    implementation = get_implementation(args.filter, "filters")

languages = implementation.languages
if languages != "All" and args.language not in languages:
    raise ValueError(f"The specified transformation is applicable only for the locales={languages}.")
evaluate(
    implementation,
    args.task_type,
    args.language,
    args.model,
    args.dataset,
    args.percentage_of_examples,
    if_filter,
    args.batch_size,
)
