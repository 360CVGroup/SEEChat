""""writer module handle writing the images to disk"""

import json
import os

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import webdataset as wds


class BufferedParquetWriter:
    """Write samples to parquet files incrementally with a buffer"""

    def __init__(self, output_file, schema, buffer_size=100):
        self.buffer_size = buffer_size
        self.schema = schema
        self._initiatlize_buffer()
        fs, output_path = fsspec.core.url_to_fs(output_file)

        self.output_fd = fs.open(output_path, "wb")
        self.parquet_writer = pq.ParquetWriter(self.output_fd, schema)

    def _initiatlize_buffer(self):
        self.current_buffer_size = 0
        self.buffer = {k: [] for k in self.schema.names}

    def _add_sample_to_buffer(self, sample):
        for k in self.schema.names:
            self.buffer[k].append(sample[k])
        self.current_buffer_size += 1

    def write(self, sample):
        if self.current_buffer_size >= self.buffer_size:
            self.flush()
        self._add_sample_to_buffer(sample)

    def flush(self):
        """Write the buffer to disk"""
        if self.current_buffer_size == 0:
            return

        df = pa.Table.from_pydict(self.buffer, self.schema)
        self.parquet_writer.write_table(df)
        self._initiatlize_buffer()

    def close(self):
        self.flush()
        if self.parquet_writer is not None:
            self.parquet_writer.close()
            self.parquet_writer = None
            self.output_fd.close()


class ParquetSampleWriter:
    """ParquetSampleWriter is a image+caption writer to parquet"""

    def __init__(
        self,
        shard_id,
        output_folder,
        save_caption,
        oom_shard_count,
        schema,
        encode_format,
    ):
        self.oom_shard_count = oom_shard_count
        self.encode_format = encode_format
        schema = schema.append(pa.field(encode_format, pa.binary()))
        shard_name = "{shard_id:0{oom_shard_count}d}".format(  # pylint: disable=consider-using-f-string
            shard_id=shard_id, oom_shard_count=oom_shard_count
        )
        output_file = f"{output_folder}/{shard_name}.parquet"
        self.buffered_parquet_writer = BufferedParquetWriter(output_file, schema, 100)
        self.save_caption = save_caption

    def write(self, img_str, key, caption, meta):
        """Keep sample in memory then write to disk when close() is called"""
        if img_str is not None:
            sample = {"key": key, self.encode_format: img_str}
            if self.save_caption:
                sample["txt"] = str(caption) if caption is not None else ""
        else:
            sample = {"key": key, self.encode_format: None}
            if self.save_caption:
                sample["txt"] = None
        sample.update(meta)
        self.buffered_parquet_writer.write(sample)

    def close(self):
        self.buffered_parquet_writer.close()


class WebDatasetSampleWriter:
    """WebDatasetSampleWriter is a image+caption writer to webdataset"""

    def __init__(
        self,
        shard_id,
        output_folder,
        save_caption,
        oom_shard_count,
        schema,
        encode_format,
        ):

        self.oom_shard_count = oom_shard_count
        shard_name = "{shard_id:0{oom_shard_count}d}".format(  # pylint: disable=consider-using-f-string
            shard_id=shard_id, oom_shard_count=oom_shard_count
        )
        self.shard_id = shard_id
        fs, output_path = fsspec.core.url_to_fs(output_folder)
        
        self.tar_fd = fs.open(f"{output_path}/{shard_name}.tar", "wb")
        self.tarwriter = wds.TarWriter(self.tar_fd)
        self.save_caption = save_caption
        self.buffered_parquet_writer = BufferedParquetWriter(output_folder + "/" + shard_name + ".parquet", schema, 100)
        self.encode_format = encode_format

    def write(self, img_str, key, caption, meta):
        """write sample to tars"""
        if img_str is not None:
            sample = {"__key__": key, self.encode_format: img_str}
            if self.save_caption:
                sample["txt"] = str(caption) if caption is not None else ""
            # some meta data may not be JSON serializable
            for k, v in meta.items():
                if isinstance(v, np.ndarray):
                    meta[k] = v.tolist()
            sample["json"] = json.dumps(meta, indent=4, ensure_ascii=False )
            self.tarwriter.write(sample)
        self.buffered_parquet_writer.write(meta)

    def close(self):
        self.buffered_parquet_writer.close()
        self.tarwriter.close()
        self.tar_fd.close()


class FilesSampleWriter:
    """FilesSampleWriter is a caption+image writer to files"""

    def __init__(
        self,
        shard_id,
        output_folder,
        save_caption,
        oom_shard_count,
        schema,
        encode_format,
    ):
        self.oom_shard_count = oom_shard_count
        shard_name = "{shard_id:0{oom_shard_count}d}".format(  # pylint: disable=consider-using-f-string
            shard_id=shard_id, oom_shard_count=oom_shard_count
        )
        self.shard_id = shard_id
        self.fs, self.subfolder = fsspec.core.url_to_fs(f"{output_folder}/{shard_name}")
        if not self.fs.exists(self.subfolder):
            self.fs.mkdir(self.subfolder)
        self.save_caption = save_caption
        self.buffered_parquet_writer = BufferedParquetWriter(output_folder + "/" + shard_name + ".parquet", schema, 100)
        self.encode_format = encode_format

    def write(self, img_str, key, caption, meta):
        """Write sample to disk"""
        if img_str is not None:
            filename = f"{self.subfolder}/{key}.{self.encode_format}"
            with self.fs.open(filename, "wb") as f:
                f.write(img_str)
            if self.save_caption:
                caption = str(caption) if caption is not None else ""
                caption_filename = f"{self.subfolder}/{key}.txt"
                with self.fs.open(caption_filename, "w") as f:
                    f.write(str(caption))

            # some meta data may not be JSON serializable
            for k, v in meta.items():
                if isinstance(v, np.ndarray):
                    meta[k] = v.tolist()
            j = json.dumps(meta, indent=4)
            meta_filename = f"{self.subfolder}/{key}.json"
            with self.fs.open(meta_filename, "w") as f:
                f.write(j)
        self.buffered_parquet_writer.write(meta)

    def close(self):
        self.buffered_parquet_writer.close()


class DummySampleWriter:
    """Does not write"""

    def __init__(self, shard_id, output_folder, save_caption, oom_shard_count, schema, encode_format):
        pass

    def write(self, img_str, key, caption, meta):
        pass

    def close(self):
        pass
