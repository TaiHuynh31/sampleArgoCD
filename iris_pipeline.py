# from kfp import dsl, compiler
# from components.preprocess import preprocess_op
# from components.train.train import train_op
# from components.evaluate.evaluate import evaluate_op
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)
#
# # Define the pipeline
# @dsl.pipeline(
#     name="Iris Training Pipeline",
#     description="A pipeline to preprocess the Iris dataset, train a model, and evaluate it."
# )
# def iris_pipeline():
#     # Step 1: Preprocess Data
#     preprocess_task = preprocess_op()
#
#     # Step 2: Train Model
#     train_task = train_op(dataset=preprocess_task.outputs['output_dataset'])
#
#     # Step 3: Evaluate Model
#     evaluate_task = evaluate_op(
#         dataset=preprocess_task.outputs['output_dataset'],
#         model=train_task.outputs['output_model']
#     )
#
# # Compile the pipeline to a YAML file
# if __name__ == '__main__':
#     pipeline_file = "iris_pipeline.yaml"
#     compiler.Compiler().compile(
#         pipeline_func=iris_pipeline,
#         package_path=pipeline_file
#     )
#
from kfp.dsl import pipeline, component, Input, Output, Dataset, Model, Artifact

@component(base_image="taihuynh31/preprocess:latest")
def preprocess_op(output_dataset: Output[Dataset]):
    pass

@component(base_image="taihuynh31/train:latest")
def train_op(dataset: Input[Dataset], output_model: Output[Model]):
    pass

@component(base_image="taihuynh31/evaluate:latest")
def evaluate_op(dataset: Input[Dataset], model: Input[Model], metrics: Output[Artifact]):
    pass

@pipeline(
    name="Iris Training Pipeline",
    description="Pipeline to preprocess, train, and evaluate Iris model."
)
def iris_pipeline():
    preprocess_task = preprocess_op()
    train_task = train_op(dataset=preprocess_task.outputs["output_dataset"])
    evaluate_task = evaluate_op(
        dataset=preprocess_task.outputs["output_dataset"],
        model=train_task.outputs["output_model"]
    )

if __name__ == "__main__":
    from kfp.compiler import Compiler
    Compiler().compile(iris_pipeline, "iris_pipeline.yaml")
