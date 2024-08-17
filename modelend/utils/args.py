import argparse


def parse_init():
    parser = argparse.ArgumentParser()
    parser = parse_preprocessing(parser)
    parser = parse_classifier(parser)
    parser = parse_t5(parser)
    parser = parse_retrain(parser)

    opt = parser.parse_args()
    return opt


def parse_preprocessing(parser):
    parser.add_argument('--table_path', type=str, default="./data/spider/test_data/tables.json")
    parser.add_argument('--mode', type=str, default="test",
                        help='trian, eval or test.')
    return parser


def parse_classifier(parser):
    parser.add_argument('--db_path', type=str, default="./data/spider/test_database",
                        help="the filepath of database.")
    parser.add_argument('--batch_size', type=int, default=1,
                        help='input batch size.')
    parser.add_argument('--device', type=str, default="0",
                        help='the id of used GPU device.')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Threshold of schema linking')
    parser.add_argument('--linking_mode', type=str, default="linking",
                        help='select linking mode , including None, linking, interaction')
    parser.add_argument('--classifier_name_or_path', type=str, default="./checkpoints/text2sql_schema_item_classifier",
                        help='''pre-trained model name.''')
    parser.add_argument('--use_contents', default=True,
                        help='whether to integrate db contents into input sequence')
    parser.add_argument('--add_fk_info', default=True,
                        help='whether to add [FK] tokens into input sequence')
    parser.add_argument('--topk_table_num', type=int, default=4,
                        help='we only remain topk_table_num tables in the ranked dataset (k_1 in the paper).')
    parser.add_argument('--topk_column_num', type=int, default=5,
                        help='we only remain topk_column_num columns for each table in the ranked dataset (k_2 in the paper).')
    parser.add_argument('--noise_rate', type=float, default=0.08,
                        help='the noise rate in the ranked training dataset (needed when the mode = "train")')
    parser.add_argument('--output_skeleton', default=False,
                        help='whether to add skeleton in the output sequence.')
    return parser


def parse_t5(parser):
    # parser.add_argument('--model_name_or_path_nat', type=str, default="./adaptive/models/filter_0.5/checkpoint-4824/", )
    parser.add_argument('--model_name_or_path', type=str, default="./checkpoints/text2sql-t5-base/checkpoint-39312", )
    # ./checkpoints/text2sql-t5-base/checkpoint-39312
    # ./checkpoints/text2sql-t5-large/checkpoint-30576
    parser.add_argument('--use_natsql', default=False,
                        help='whether to use natsql')
    parser.add_argument('--num_beams', type=int, default=4,
                        help='beam size in model.generate() function.')
    parser.add_argument('--tables_for_natsql', type=str, default="./data/spider/tables_for_natsql.json",
                        help='file path of tables_for_natsql.json.')
    parser.add_argument('--num_return_sequences', type=int, default=4,
                        help='the number of returned sequences in model.generate() function (num_return_sequences <= num_beams).')
    parser.add_argument('--predict_result', type=str,
                        default='target/output_result/test/filter_0.5/checkpoint-4824-excellent.txt',
                        help='save path of best fine-tuned text2sql model.')

    return parser


def parse_retrain(parser):
    parser.add_argument('--train_file', type=str, default="./data/train_data/resdsql_train_spider.json",
                        help="the filepath of database.")
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed.')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                        help='learning rate.')
    parser.add_argument('--gradient_descent_step', type=int, default=4,
                        help='perform gradient descent per "gradient_descent_step" steps.')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='gamma parameter in the focal loss. Recommended: [0.0-2.0].')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='alpha parameter in the focal loss. Must between [0.0 -1.0].')
    parser.add_argument('--epochs', type=int, default=6,
                        help='training epochs.')
    parser.add_argument('--save_path', type=str, default="adaptive/models/filter_0.9/",
                        help='save path of best fine-tuned text2sql model.')
    parser.add_argument('--db_list_path', type=str, default="data/train_data/db_list0.9.json",
                        help='save path of best fine-tuned text2sql model.')
    parser.add_argument('--use_adafactor', type=bool, default=True,
                        help='whether to use adafactor optimizer.')

    return parser
