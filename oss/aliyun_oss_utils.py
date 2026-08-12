"""
阿里云 OSS 读写工具类

使用前需要:
1. 开通阿里云 OSS 服务: https://www.aliyun.com/product/oss
2. 创建 Bucket
3. 获取 AccessKey:
   - 登录阿里云控制台 -> 右上角头像 -> AccessKey 管理
   - 建议使用 RAM 子账号的 AccessKey（更安全，可限制权限）
   - 需要授予 OSS 相关权限（AliyunOSSFullAccess 或自定义策略）
4. 安装依赖: pip install oss2

所需配置项:
- OSS_ACCESS_KEY_ID: AccessKey ID
- OSS_ACCESS_KEY_SECRET: AccessKey Secret
- OSS_ENDPOINT: OSS 服务节点，如 https://oss-cn-hangzhou.aliyuncs.com
- OSS_BUCKET_NAME: Bucket 名称
"""

import os
import oss2
from typing import Optional, BinaryIO


class OSSClient:
    """阿里云 OSS 客户端封装"""

    def __init__(
        self,
        access_key_id: Optional[str] = None,
        access_key_secret: Optional[str] = None,
        endpoint: Optional[str] = None,
        bucket_name: Optional[str] = None,
    ):
        """
        初始化 OSS 客户端。
        参数优先使用传入值，否则从环境变量读取:
          - OSS_ACCESS_KEY_ID
          - OSS_ACCESS_KEY_SECRET
          - OSS_ENDPOINT
          - OSS_BUCKET_NAME
        """
        self.access_key_id = access_key_id or os.getenv("OSS_ACCESS_KEY_ID")
        self.access_key_secret = access_key_secret or os.getenv("OSS_ACCESS_KEY_SECRET")
        self.endpoint = endpoint or os.getenv("OSS_ENDPOINT")
        self.bucket_name = bucket_name or os.getenv("OSS_BUCKET_NAME")

        if not all([self.access_key_id, self.access_key_secret, self.endpoint, self.bucket_name]):
            raise ValueError(
                "缺少必要配置，请提供 access_key_id, access_key_secret, endpoint, bucket_name，"
                "或设置对应环境变量: OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET, OSS_ENDPOINT, OSS_BUCKET_NAME"
            )

        auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        self.bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)

    # ==================== 上传 ====================

    def put_object(self, object_key: str, data: bytes) -> str:
        """
        上传字节数据到 OSS。

        :param object_key: 对象路径（含目录前缀），如 "images/photo.jpg"
        :param data: 字节数据
        :return: 上传后的 ETag
        """
        result = self.bucket.put_object(object_key, data)
        print(f"[OSS] 上传成功: {object_key} (ETag: {result.etag})")
        return result.etag

    def put_object_from_file(self, object_key: str, local_file_path: str) -> str:
        """
        从本地文件上传到 OSS。

        :param object_key: OSS 上的对象路径
        :param local_file_path: 本地文件路径
        :return: 上传后的 ETag
        """
        result = self.bucket.put_object_from_file(object_key, local_file_path)
        print(f"[OSS] 文件上传成功: {local_file_path} -> {object_key}")
        return result.etag

    def put_object_from_stream(self, object_key: str, stream: BinaryIO) -> str:
        """
        从流中上传数据到 OSS。

        :param object_key: OSS 上的对象路径
        :param stream: 可读的二进制流
        :return: 上传后的 ETag
        """
        result = self.bucket.put_object(object_key, stream)
        print(f"[OSS] 流上传成功: -> {object_key}")
        return result.etag

    # ==================== 下载 ====================

    def get_object(self, object_key: str) -> bytes:
        """
        下载对象为字节数据。

        :param object_key: OSS 上的对象路径
        :return: 对象内容（bytes）
        """
        result = self.bucket.get_object(object_key)
        data = result.read()
        result.close()
        print(f"[OSS] 下载成功: {object_key} ({len(data)} bytes)")
        return data

    def get_object_to_file(self, object_key: str, local_file_path: str) -> None:
        """
        下载对象到本地文件。

        :param object_key: OSS 上的对象路径
        :param local_file_path: 本地保存路径
        """
        self.bucket.get_object_to_file(object_key, local_file_path)
        print(f"[OSS] 下载成功: {object_key} -> {local_file_path}")

    def get_object_as_stream(self, object_key: str) -> BinaryIO:
        """
        获取对象的可读流（调用方需负责关闭）。

        :param object_key: OSS 上的对象路径
        :return: 可读的二进制流
        """
        result = self.bucket.get_object(object_key)
        print(f"[OSS] 获取流成功: {object_key}")
        return result

    # ==================== 查询与删除 ====================

    def object_exists(self, object_key: str) -> bool:
        """检查对象是否存在。"""
        return self.bucket.object_exists(object_key)

    def delete_object(self, object_key: str) -> None:
        """删除指定对象。"""
        self.bucket.delete_object(object_key)
        print(f"[OSS] 已删除: {object_key}")

    def delete_objects(self, object_keys: list[str]) -> None:
        """批量删除对象。"""
        self.bucket.batch_delete_objects(object_keys)
        print(f"[OSS] 批量删除 {len(object_keys)} 个对象")

    def list_objects(self, prefix: str = "", max_keys: int = 100) -> list[str]:
        """
        列出指定前缀下的对象。

        :param prefix: 前缀过滤，如 "images/"
        :param max_keys: 最大返回数量
        :return: 对象路径列表
        """
        keys = []
        for obj in oss2.ObjectIterator(self.bucket, prefix=prefix, max_keys=max_keys):
            keys.append(obj.key)
        print(f"[OSS] 列出 {len(keys)} 个对象 (prefix={prefix!r})")
        return keys

    # ==================== URL 与预签名 ====================

    def get_object_url(self, object_key: str) -> str:
        """获取对象的公开访问 URL（Bucket 需为公开读权限）。"""
        url = f"https://{self.bucket_name}.{self.endpoint.replace('https://', '')}/{object_key}"
        return url

    def sign_url(self, object_key: str, expires: int = 3600) -> str:
        """
        生成带签名的临时访问 URL。

        :param object_key: OSS 上的对象路径
        :param expires: URL 有效期（秒），默认 1 小时
        :return: 签名 URL
        """
        url = self.bucket.sign_url("GET", object_key, expires)
        print(f"[OSS] 签名 URL 已生成: {object_key} (有效期 {expires}s)")
        return url


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 方式一：通过环境变量配置（推荐）
    # export OSS_ACCESS_KEY_ID="your-access-key-id"
    # export OSS_ACCESS_KEY_SECRET="your-access-key-secret"
    # export OSS_ENDPOINT="https://oss-cn-hangzhou.aliyuncs.com"
    # export OSS_BUCKET_NAME="your-bucket-name"

    # 方式二：直接传参
    client = OSSClient(
        access_key_id="your-access-key-id",
        access_key_secret="your-access-key-secret",
        endpoint="https://oss-cn-hangzhou.aliyuncs.com",
        bucket_name="your-bucket-name",
    )

    # --- 上传示例 ---
    # 上传字节数据
    client.put_object("test/hello.txt", b"Hello, Alibaba Cloud OSS!")

    # 上传本地文件
    # client.put_object_from_file("test/image.png", "/path/to/local/image.png")

    # --- 下载示例 ---
    data = client.get_object("test/hello.txt")
    print(f"下载内容: {data.decode('utf-8')}")

    # 下载到本地文件
    # client.get_object_to_file("test/image.png", "/path/to/save/image.png")

    # --- 查询示例 ---
    print(f"文件存在: {client.object_exists('test/hello.txt')}")
    files = client.list_objects(prefix="test/")
    print(f"文件列表: {files}")

    # --- 签名 URL ---
    signed_url = client.sign_url("test/hello.txt", expires=7200)
    print(f"临时访问 URL: {signed_url}")

    # --- 删除示例 ---
    # client.delete_object("test/hello.txt")
