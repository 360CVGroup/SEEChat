"""Img2dataset"""

from typing import List, Optional
# import logging
import fsspec
# import sys
import os


from resizer import Resizer


from reader import Reader
from downloader import Downloader
from writer import WebDatasetSampleWriter
from distributor import multiprocessing_distributor# , pyspark_distributor

# # 参数校验
def arguments_validator(params):
    """Validate the arguments"""
    if params["compute_hash"] not in [None, "md5", "sha256", "sha512"]:
        hash_type = params["compute_hash"]
        raise ValueError(f"Unsupported hash to compute: {hash_type}")

    if params["verify_hash"] is not None:
        _, verify_hash_type = params["verify_hash"]
        if verify_hash_type != params["compute_hash"]:
            raise ValueError(
                "verify_hash and compute_hash must be the same "
                f"but got {verify_hash_type} and {params['compute_hash']}"
            )

    if params["save_additional_columns"] is not None:
        save_additional_columns_set = set(params["save_additional_columns"])

        forbidden_columns = set(
            [
                "key",
                "caption",
                "url",
                "width",
                "height",
                "original_width",
                "original_height",
                "status",
                "error_message",
                "exif",
                "md5",
                "sha256",
                "sha512",
            ]
        )
        intersection = save_additional_columns_set.intersection(forbidden_columns)
        if intersection:
            raise ValueError(
                f"You cannot use in save_additional_columns the following columns: {intersection}."
                + "img2dataset reserves these columns for its own use. Please remove them from save_additional_columns."
            )


def download(
    url_list: str,
    image_size: int = 256,
    output_folder: str = "images",
    processes_count: int = 1,
    resize_mode: str = "border",
    resize_only_if_bigger: bool = False,
    upscale_interpolation: str = "lanczos",
    downscale_interpolation: str = "area",
    encode_quality: int = 95,
    encode_format: str = "jpg",

    skip_reencode: bool = False,
    output_format: str = "files",
    input_format: str = "txt",
    url_col: str = "url",
    caption_col: Optional[str] = None,
    # bbox_col: Optional[str] = None,

    thread_count: int = 256,
    number_sample_per_shard: int = 10000,
    extract_exif: bool = True,
    save_additional_columns: Optional[List[str]] = None,
    timeout: int = 10,
    enable_wandb: bool = False,
    wandb_project: str = "img2dataset",
    oom_shard_count: int = 5,
    compute_hash: Optional[str] = "sha256",
    verify_hash: Optional[List[str]] = None,
    distributor: str = "multiprocessing",
    subjob_size: int = 1000,
    retries: int = 0,

    disable_all_reencoding: bool = False,
    min_image_size: int = 0,
    max_image_area: float = float("inf"),
    max_aspect_ratio: float = float("inf"),
    incremental_mode: str = "incremental",
    max_shard_retry: int = 1,
    user_agent_token: Optional[str] = None,
    # disallowed_header_directives: Optional[List[str]] = None,
    ):
    # """Download is the main entry point of img2dataset, it uses multiple processes and download multiple files"""
    # if disallowed_header_directives is None:
    #     disallowed_header_directives = ["noai", "noimageai", "noindex", "noimageindex"]
    # if len(disallowed_header_directives) == 0:
    #     disallowed_header_directives = None

    # 校验参数
    config_parameters = dict(locals())
    print(config_parameters)
    arguments_validator(config_parameters)

    def make_path_absolute(path):
        fs, p = fsspec.core.url_to_fs(path)
        if fs.protocol == "file":
            return os.path.abspath(p)
        return path

    output_folder = make_path_absolute(output_folder)
    url_list = make_path_absolute(url_list)

    tmp_path = output_folder + "/_tmp"
    fs, tmp_dir = fsspec.core.url_to_fs(tmp_path)
    if not fs.exists(tmp_dir):
        fs.mkdir(tmp_dir)

    save_caption = caption_col is not None
    

    fs, output_path = fsspec.core.url_to_fs(output_folder)
    print(f"fs:{fs}, output_path:{output_path}")

    # 是否断点续传 或者覆盖
    if not fs.exists(output_path):
        fs.mkdir(output_path)
        done_shards = set()
    else:
        if incremental_mode == "incremental":
            done_shards = set(int(x.split("/")[-1].split("_")[0]) for x in fs.glob(output_path + "/*.json"))
        elif incremental_mode == "overwrite":
            fs.rm(output_path, recursive=True)
            fs.mkdir(output_path)
            done_shards = set()
        else:
            raise ValueError(f"Unknown incremental mode {incremental_mode}")


    if verify_hash is not None:
        verify_hash_col, verify_hash_type = verify_hash
    else:
        verify_hash_col = None
        verify_hash_type = None

    if output_format == "webdataset":
        sample_writer_class = WebDatasetSampleWriter
    else:
        raise ValueError(f"Invalid output format {output_format}")

    print("Starting the downloading of this file")
    if distributor == "multiprocessing":
        distributor_fn = multiprocessing_distributor
    else:
        raise ValueError(f"Distributor {distributor} not supported")

    reader = Reader(
        url_list,
        input_format,
        url_col,
        caption_col,
        verify_hash_col,
        verify_hash_type,
        save_additional_columns,
        number_sample_per_shard,
        done_shards,
        tmp_path,
    )

    resizer = Resizer(
        image_size=image_size,
        resize_mode=resize_mode,
        resize_only_if_bigger=resize_only_if_bigger,
        upscale_interpolation=upscale_interpolation,
        downscale_interpolation=downscale_interpolation,
        encode_quality=encode_quality,
        encode_format=encode_format,
        skip_reencode=skip_reencode,
        disable_all_reencoding=disable_all_reencoding,
        min_image_size=min_image_size,
        max_image_area=max_image_area,
        max_aspect_ratio=max_aspect_ratio,
        blurrer=None,
    )

    downloader = Downloader(
        sample_writer_class=sample_writer_class,
        resizer=resizer,
        thread_count=thread_count,
        save_caption=save_caption,
        extract_exif=extract_exif,
        output_folder=output_folder,
        column_list=reader.column_list,
        timeout=timeout,
        number_sample_per_shard=number_sample_per_shard,
        oom_shard_count=oom_shard_count,
        compute_hash=compute_hash,
        verify_hash_type=verify_hash_type,
        encode_format=encode_format,
        retries=retries,
        user_agent_token=user_agent_token,
        # disallowed_header_directives=disallowed_header_directives,
        # blurring_bbox_col=bbox_col,
    )

    distributor_fn(
        processes_count,
        downloader,
        reader,
        subjob_size,
        max_shard_retry,
    )

    # 删除临时目录
    # fs.rm(tmp_dir, recursive=True)


def main():
    # fire.Fire(download)
    url_list = "/path_to/dataset/R2D2dataset/select_caption.csv"
    input_format = "csv"
    url_col = "url"
    caption_col = "caption"

    output_folder = "/path_to/dataset/R2D2dataset/my_data800w"
    output_format = "webdataset"
    image_size = 256
    processes_count = 32
    thread_count = 256

    download( url_list = url_list, input_format = input_format, url_col = url_col, \
    caption_col = caption_col,  output_folder = output_folder, \
    output_format = output_format, image_size = image_size, processes_count = processes_count ,thread_count = thread_count )


if __name__ == "__main__":
    main()
