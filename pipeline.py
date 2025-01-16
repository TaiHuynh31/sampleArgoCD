import kfp
from kfp import dsl
from components.preprocess import preprocess_op
from components.train import train_op
from components.evaluate import evaluate_op

# Define the pipeline
@dsl.pipeline(
    name="Iris Training Pipeline",
    description="A pipeline to preprocess the Iris dataset, train a model, and evaluate it."
)
def iris_pipeline():
    # Step 1: Preprocess Data
    preprocess_task = preprocess_op()

    # Step 2: Train Model
    train_task = train_op(dataset=preprocess_task.outputs['output_dataset'])

    # Step 3: Evaluate Model
    evaluate_task = evaluate_op(
        dataset=preprocess_task.outputs['output_dataset'],
        model=train_task.outputs['output_model']
    )

# Compile the pipeline to a YAML file
if __name__ == '__main__':
    kfp.compiler.Compiler().compile(
        pipeline_func=iris_pipeline,
        package_path="iris_pipeline.yaml"
    )