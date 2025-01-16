import kfp
from kfp import dsl, compiler
from components.preprocess import preprocess_op
from components.train import train_op
from components.evaluate import evaluate_op
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
    pipeline_file = "iris_pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=iris_pipeline,
        package_path=pipeline_file
    )

    print(f"Pipeline written to {pipeline_file}")
    # run the pipeline sdk to run the pipeline
    client = kfp.Client()
    run = client.create_run_from_pipeline_package(
        'pipeline/image_classification_pipeline.yaml',
        arguments={},
        experiment_name='image-classification-experiment'
    )

