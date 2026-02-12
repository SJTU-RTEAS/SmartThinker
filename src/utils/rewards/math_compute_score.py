from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from math_verify.metric import math_metric
from math_verify.errors import TimeoutException
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rewards.norm_answer import normalize_latex

def math_compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    timeout_score: float = 0,
    extra_info: dict = None,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    
    end_tag = '</think>'
    if end_tag not in solution_str:
        return 0.0

    completion_after_think = solution_str.split(end_tag)[-1]
    completion_after_think = normalize_latex(completion_after_think)
    ground_truth = normalize_latex(ground_truth)
    # if ground_truth is not surrounded by $...$, add them
    for mat in [('$', '$'), ('\\(', '\\)'), ('\\[', '\\]')]:
        if ground_truth.startswith(mat[0]) and ground_truth.endswith(mat[1]):
            ground_truth = ground_truth[len(mat[0]):-len(mat[1])].strip()
            break
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [solution_str])
    except Exception:
        ret_score = 0.0
    except TimeoutException:
        ret_score = timeout_score
        
    return ret_score