



from .vg import vg_evaluation


def evaluate(predictions,targets,**kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(
        predictions=predictions, targets=targets,**kwargs
    )


    return vg_evaluation(**args)
