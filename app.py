import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gradio as gr
from tugas_akhir_module import predict_image, train_update, evaluate_model, change_dropdown, list_model, get_model_hash
import argparse

with gr.Blocks(css="footer {visibility: hidden}" ) as demo:
	model_dropdown = gr.Dropdown(list_model(), label="Select Model")
	tf_model = gr.State()
	with gr.Column(visible = False) as gr_form:
		with gr.Tab("Klasifikasi Gambar"):
			gr.Markdown(
					"""
					<h1 style="text-align: center;">Klasifikasi Gambar</h1>
					""")
			with gr.Row():
				klasifikasi_input = gr.Image(label="Input")
				with gr.Column():
					klasifikasi_label = gr.Label(label="Output", num_top_classes=2)
					with gr.Accordion("Gambar Setelah Proses Rescale", open=False):
						with gr.Row():
							klasifikasi_img_resized = gr.Image()
			klasifikasi_submit = gr.Button("Submit")
			klasifikasi_submit.click(predict_image, inputs = [tf_model, klasifikasi_input], outputs = [klasifikasi_label, klasifikasi_img_resized])

		with gr.Tab("Training Model"):
			gr.Markdown(
					"""
					<h1 style="text-align: center;">Training Model</h1>
					""")
			with gr.Row():
				with gr.Column():
					training_data_train = gr.Textbox(label="Path Data Train", info="Enter Path to Train Dataset Folder", placeholder="/content/chest_xray/train")
					training_save_model  = gr.Textbox(label="Model Name", info="Enter Model Name", placeholder="model.h5")
					training_data_val = gr.Textbox(label="Path Data Val", info="Enter Path to Validation Dataset Folder", placeholder="/content/chest_xray/val")
					training_epochs = gr.Slider(minimum=2, maximum=10000, label="Training Epochs", step=1)
					training_lr = gr.Slider(minimum=10**-6, maximum=10**-2, label="Learning Rate", step=10**-7)
					training_fn_epochs = gr.Slider(minimum=2, maximum=10000, label="Finetune Epochs", step=1)
					training_fn_lr = gr.Slider(minimum=10**-6, maximum=10**-2, label="Finetune Learning Rate", step=10**-7)

				with gr.Column():
					training_plot = gr.Plot()

			training_submit = gr.Button("Submit")
			training_submit.click(train_update, inputs=[tf_model, training_data_train, training_data_val, training_save_model, training_epochs, training_fn_epochs, training_lr, training_fn_lr] , outputs = [tf_model, training_plot, model_dropdown])

		with gr.Tab("Evaluasi Model"):
			gr.Markdown(
					"""
					<h1 style="text-align: center;">Evaluasi Model</h1>
					""")
			with gr.Row():
				with gr.Column():
					evaluasi_data_test = gr.Textbox(label="Path Data Test", info="Enter Path to Test Dataset Folder", placeholder="/content/chest_xray/test")
				with gr.Column():
					evaluasi_plot = gr.Plot()
					evaluasi_label = gr.Label(label="Output", show_label=False)
			evaluasi_submit = gr.Button("Submit")
			evaluasi_submit.click(evaluate_model, inputs = [tf_model, evaluasi_data_test], outputs = [evaluasi_plot, evaluasi_label] )
		
		with gr.Tab("Calculate Model Hash"):
			gr.Markdown(
					"""
					<h1 style="text-align: center;">Calculate Model Hash</h1>
					""")
			with gr.Row():
				hash_submit = gr.Button("Calculate Hash")
				hash_output = gr.Textbox(label="Model Hash")
				hash_submit.click(get_model_hash, inputs = model_dropdown, outputs = hash_output)

	model_dropdown.change(change_dropdown, inputs=model_dropdown, outputs=[tf_model, gr_form, model_dropdown])

argParser = argparse.ArgumentParser()
argParser.add_argument("-s", "--share", type=bool, help="Provide shareable link", default=False)

args = argParser.parse_args()

demo.queue().launch(debug=True, share=args.share)