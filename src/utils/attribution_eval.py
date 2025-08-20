# Imports

# Quantus Imports
import quantus


# Quantus Metrics
class QuantusMetrics:
    def __init__(self, metric_name, metric_kwargs=None):
        self.metric_name = metric_name
        self.metric_kwargs = metric_kwargs if metric_kwargs is not None else {}
        return


    def compute_metric(self):
        pass



# Faithfulness: Quantifies to what extent explanations follow the predictive behaviour of the model 
# (asserting that more important features play a larger role in model outcomes)
class QuantusFaithfulness(QuantusMetrics):
    def __init__(self, metric_name, metric_kwargs=None):
        super().__init__(metric_name, metric_kwargs)
        assert metric_name in (
            "FaithfulnessCorrelation",
            "FaithfulnessEstimate",
            "Monotonicity",
            "MonotonicityCorrelation",
            "PixelFlipping",
            "RegionPerturbation",
            "Selectivity",
            "SensitivityN",
            "IROF",
            "Infidelity",
            "ROAD",
            "Sufficiency"
        ), f"Invalid metric name: {metric_name}"

        if metric_name == "FaithfulnessCorrelation":
            self.metric = quantus.FaithfulnessCorrelation(**self.metric_kwargs)
        elif metric_name == "FaithfulnessEstimate":
            self.metric = quantus.FaithfulnessEstimate(**self.metric_kwargs)
        elif metric_name == "Monotonicity":
            self.metric = quantus.Monotonicity(**self.metric_kwargs)
        elif metric_name == "MonotonicityCorrelation":
            self.metric = quantus.MonotonicityCorrelation(**self.metric_kwargs)
        elif metric_name == "PixelFlipping":
            self.metric = quantus.PixelFlipping(**self.metric_kwargs)
        elif metric_name == "RegionPerturbation":
            self.metric = quantus.RegionPerturbation(**self.metric_kwargs)
        elif metric_name == "Selectivity":
            self.metric = quantus.Selectivity(**self.metric_kwargs)
        elif metric_name == "SensitivityN":
            self.metric = quantus.SensitivityN(**self.metric_kwargs)
        elif metric_name == "IROF":
            self.metric = quantus.IROF(**self.metric_kwargs)
        elif metric_name == "Infidelity":
            self.metric = quantus.Infidelity(**self.metric_kwargs)
        elif metric_name == "ROAD":
            self.metric = quantus.ROAD(**self.metric_kwargs)
        else:
            self.metric = quantus.Sufficiency(**self.metric_kwargs)
        
        return


    # Method: Compute the metric for a batch of data
    def compute_metric(self, model, x_batch, y_batch, a_batch, device):
        score = self.metric(model=model, x=x_batch, y=y_batch, a=a_batch, device=device)
        return score



# Robustness: Measures to what extent explanations are stable when subject to slight perturbations of the input, assuming that model 
# output approximately stayed the same
class QuantusRobustness(QuantusMetrics):
    def __init__(self, metric_name, metric_kwargs=None):
        super().__init__(metric_name, metric_kwargs)
        assert metric_name in (
            "LocalLipschitzEstimate",
            "MaxSensitivity",
            "AvgSensitivity",
            "Continuity",
            "Consistency",
            "RelativeInputStability",
            "RelativeOutputStability",
            "RelativeRepresentationStability"
        ), f"Invalid metric name: {metric_name}"

        if metric_name == "LocalLipschitzEstimate":
            self.metric = quantus.LocalLipschitzEstimate(**self.metric_kwargs)
        elif metric_name == "MaxSensitivity":
            self.metric = quantus.MaxSensitivity(**self.metric_kwargs)
        elif metric_name == "AvgSensitivity":
            self.metric = quantus.AvgSensitivity(**self.metric_kwargs)
        elif metric_name == "Continuity":
            self.metric = quantus.Continuity(**self.metric_kwargs)
        elif metric_name == "Consistency":
            self.metric = quantus.Consistency(**self.metric_kwargs)
        elif metric_name == "RelativeInputStability":
            self.metric = quantus.RelativeInputStability(**self.metric_kwargs)
        elif metric_name == "RelativeOutputStability":
            self.metric = quantus.RelativeOutputStability(**self.metric_kwargs)
        else:
            self.metric = quantus.RelativeRepresentationStability(**self.metric_kwargs)
        
        return


    # Method: Compute the metric for a batch of data
    def compute_metric(self, model, x_batch, y_batch, a_batch, device):
        score = self.metric(model=model, x=x_batch, y=y_batch, a=a_batch, device=device)
        return score



# Localisation: Tests if the explainable evidence is centred around a region of interest (RoI) which may be defined around an object 
# by a bounding box, a segmentation mask or, a cell within a grid
class QuantusLocalisation(QuantusMetrics):
    def __init__(self, metric_name, metric_kwargs=None):
        super().__init__(metric_name, metric_kwargs)
        assert metric_name in (
            "PointingGame",
            "AttributionLocalisation",
            "TopKIntersection",
            "RelevanceRankAccuracy",
            "RelevanceMassAccuracy",
            "AUC"
        ), f"Invalid metric name: {metric_name}"

        if metric_name == "PointingGame":
            self.metric = quantus.PointingGame(**self.metric_kwargs)
        elif metric_name == "AttributionLocalisation":
            self.metric = quantus.AttributionLocalisation(**self.metric_kwargs)
        elif metric_name == "TopKIntersection":
            self.metric = quantus.TopKIntersection(**self.metric_kwargs)
        elif metric_name == "RelevanceRankAccuracy":
            self.metric = quantus.RelevanceRankAccuracy(**self.metric_kwargs)
        elif metric_name == "RelevanceMassAccuracy":
            self.metric = quantus.RelevanceMassAccuracy(**self.metric_kwargs)
        else:
            self.metric = quantus.AUC(**self.metric_kwargs)
        
        return


    # Method: Compute the metric for a batch of data
    def compute_metric(self, model, x_batch, y_batch, a_batch, device):
        score = self.metric(model=model, x=x_batch, y=y_batch, a=a_batch, device=device)
        return score



# Complexity: Captures to what extent explanations are concise i.e., that few features are used to explain a model prediction
class QuantusComplexity(QuantusMetrics):
    def __init__(self, metric_name, metric_kwargs=None):
        super().__init__(metric_name, metric_kwargs)
        assert metric_name in (
            "Sparseness",
            "Complexity",
            "EffectiveComplexity"
        ), f"Invalid metric name: {metric_name}"

        if metric_name == "Sparseness":
            self.metric = quantus.Sparseness(**self.metric_kwargs)
        elif metric_name == "Complexity":
            self.metric = quantus.Complexity(**self.metric_kwargs)
        else:
            self.metric = quantus.EffectiveComplexity(**self.metric_kwargs)

        return


    # Method: Compute the metric for a batch of data
    def compute_metric(self, model, x_batch, y_batch, a_batch, device):
        score = self.metric(model=model, x=x_batch, y=y_batch, a=a_batch, device=device)
        return score



# Randomisation (Sensitivity): Tests to what extent explanations deteriorate as inputs to the evaluation problem 
# e.g., model parameters are increasingly randomised
class QuantusRandomisation(QuantusMetrics):
    def __init__(self, metric_name, metric_kwargs=None):
        super().__init__(metric_name, metric_kwargs)
        assert metric_name in (
            "ModelParameterRandomisation",
            "RandomLogit"
        ), f"Invalid metric name: {metric_name}"

        if metric_name == "ModelParameterRandomisation":
            self.metric = quantus.ModelParameterRandomisation(**self.metric_kwargs)
        else:
            self.metric = quantus.RandomLogit(**self.metric_kwargs)

        return


    # Method: Compute the metric for a batch of data
    def compute_metric(self, model, x_batch, y_batch, a_batch, device):
        score = self.metric(model=model, x=x_batch, y=y_batch, a=a_batch, device=device)
        return score


# Axiomatic: Assesses if explanations fulfil certain axiomatic properties
class QuantusAxiomatic(QuantusMetrics):
    def __init__(self, metric_name, metric_kwargs=None):
        super().__init__(metric_name, metric_kwargs)
        assert metric_name in (
            "Completeness",
            "NonSensitivity",
            "InputInvariance"
        ), f"Invalid metric name: {metric_name}"

        if metric_name == "Completeness":
            self.metric = quantus.Completeness(**self.metric_kwargs)
        elif metric_name == "NonSensitivity":
            self.metric = quantus.NonSensitivity(**self.metric_kwargs)
        else:
            self.metric = quantus.InputInvariance(**self.metric_kwargs)

        return


    # Method: Compute the metric for a batch of data
    def compute_metric(self, model, x_batch, y_batch, a_batch, device):
        score = self.metric(model=model, x=x_batch, y=y_batch, a=a_batch, device=device)
        return score