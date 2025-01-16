import kfp
client = kfp.Client()
client.create_run_from_pipeline_package(
    pipeline_file='iris_pipeline.yaml',
    arguments={}
)