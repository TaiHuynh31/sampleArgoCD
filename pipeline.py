import kfp
from kfp import dsl
from components.preprocess import preprocess_op
from components.train import train_op
from components.evaluate import evaluate_op

# Define the pipeline
@dsl.pipeline(
    name="AI Training Pipeline",
    description="A pipeline to preprocess data, train a model, and evaluate it."
)
def image_classification_pipeline():
    # Step 1: Preprocess Data
    preprocess_task = preprocess_op()

    # Step 2: Train Model
    train_task = train_op(dataset = preprocess_task.output)

    # Step 3: Evaluate Model
    evaluate_task = evaluate_op(dataset = preprocess_task.output, model = train_task.output)

# Compile the pipeline to a YAML file
kfp.compiler.Compiler().compile(
    pipeline_func=image_classification_pipeline,
    package_path="image_classification_pipeline.yaml"  # Specify YAML output
)