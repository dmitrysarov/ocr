По мере проведения экспериментов с обечением модели чтения текстов с картинки файлы обростали различными префиксами.
Multiprocessing - соответствует тому моменту когда в пайплайн был добавлено быстрое "скармливание" картинок в parallel fashon

первые эксперименты с моделью и receptive field --- resnet18_placeholders.ipynb 

версия использующая синтетику --- resnet18_dataset_v2.ipynb

проверка работы (получение метрик) на реальных данных --- resnet18_dataset_v3_python_multiprocessing-get_prediction_of_real_data.ipynb

проверка варианта с не использованием beamsearch --- resnet18_dataset_v3_python_multiprocessing-get_prediction_of_real_data_argmax_instead_beansearch.ipynb

разделение строк на общие и содержание только цифры, точку и запятую --- resnet18_dataset_v3_python_multiprocessing-get_prediction_of_real_data_sep_num_alfa.ipynb

resnet18_dataset_v3_python_multiprocessing.ipynb

введение падения скорости --- resnet18_dataset_v3_python_multiprocessing_validate_real-LR_decay.ipynb

обучение(finetune) на реальных данных --- resnet18_dataset_v3_python_multiprocessing_validate_real-LR_decay_learn_real.ipynb

валидация на реальном датасете --- resnet18_dataset_v3_python_multiprocessing_validate_real.ipynb

проверка на оверфит --- resnet18_dataset_v3_python_overfit_one_image.ipynb

resnet18_dataset_v3_tf_mult_proc.ipynb
