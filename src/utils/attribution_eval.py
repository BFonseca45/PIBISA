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



# Faithfulness: quantifies to what extent explanations follow the predictive behaviour of the model 
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
        )

        if metric_name == "FaithfulnessCorrelation":
            self.metric = quantus.FaithfulnessCorrelation(**self.metric_kwargs)
        
        return

    def compute_metric(self, model, x_batch, y_batch, a_batch, device):
        score = self.metric(model=model, x=x_batch, y=y_batch, a=a_batch, device=device)
        return score



# Robustness: measures to what extent explanations are stable when subject to slight perturbations of the input, assuming that model 
# output approximately stayed the same



# Localisation: tests if the explainable evidence is centred around a region of interest (RoI) which may be defined around an object 
# by a bounding box, a segmentation mask or, a cell within a grid



# Complexity: captures to what extent explanations are concise i.e., that few features are used to explain a model prediction



# Randomisation (Sensitivity): tests to what extent explanations deteriorate as inputs to the evaluation problem 
# e.g., model parameters are increasingly randomised



# Axiomatic: assesses if explanations fulfil certain axiomatic properties