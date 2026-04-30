from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./messages.db"

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)

    # 基础标识
    group_id = Column(String, nullable=True, index=True)
    user_id = Column(String, nullable=True, index=True)
    message_id = Column(String, nullable=True, index=True)

    # 原始与清洗文本
    raw_message = Column(Text, nullable=True)
    cleaned_message = Column(Text, nullable=True)

    # 消息描述字段
    message_type = Column(String, nullable=True)      # text / image / file / face / mixed / unknown
    sub_type = Column(String, nullable=True)          # normal / reply / animation_face / etc.

    is_reply = Column(Boolean, default=False)
    reply_target_id = Column(String, nullable=True)

    has_at = Column(Boolean, default=False)
    mentioned_users = Column(Text, nullable=True)     # JSON字符串，例如 ["123","456"]

    has_image = Column(Boolean, default=False)
    has_file = Column(Boolean, default=False)
    has_face = Column(Boolean, default=False)         # 普通表情
    has_animation = Column(Boolean, default=False)    # 动画表情

    # 图片资源
    image_urls = Column(Text, nullable=True)          # JSON字符串
    image_files = Column(Text, nullable=True)         # JSON字符串
    image_sizes = Column(Text, nullable=True)         # JSON字符串

    # 文件资源
    file_urls = Column(Text, nullable=True)           # JSON字符串
    file_names = Column(Text, nullable=True)          # JSON字符串
    file_sizes = Column(Text, nullable=True)          # JSON字符串

    # 表情资源
    face_ids = Column(Text, nullable=True)            # JSON字符串
    animation_files = Column(Text, nullable=True)     # JSON字符串
    animation_urls = Column(Text, nullable=True)      # JSON字符串

    # 兜底资源字段：完整保存提取出的资源信息
    resource_json = Column(Text, nullable=True)       # JSON字符串

    # 原始消息段，后续做更精确分析时很有用
    message_segments_json = Column(Text, nullable=True)

    # 时间
    received_at = Column(String, nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)