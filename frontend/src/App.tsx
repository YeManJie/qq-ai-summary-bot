import { useMemo, useState } from "react";
import {
  Button,
  Card,
  DatePicker,
  Form,
  Input,
  List,
  Tag,
  Typography,
  message,
  Spin,
  Space,
  Divider,
} from "antd";
import dayjs, { Dayjs } from "dayjs";

const { Title, Text, Paragraph } = Typography;

type Cluster = {
  cluster_id?: string;
  slice_ids?: string[];
  message_ids: string[];
  reason?: string;
};

type Classification = {
  cluster_id?: string;
  category?: string;
  reason?: string;
};

type TagResult = {
  cluster_id?: string;
  tags?: string[];
  reason?: string;
};

type PipelineResponse = {
  ok: boolean;
  group_id?: string;
  start_time?: string;
  end_time?: string;
  count?: number;
  cluster_result?: {
    parsed_json?: {
      clusters?: Cluster[];
    };
  };
  classification_result?: {
    parsed_json?: {
      cluster_classification_results?: Classification[];
    };
  };
  tag_result?: {
    parsed_json?: {
      cluster_tag_results?: TagResult[];
    };
  };
  [key: string]: unknown;
};

type SummaryResponse = {
  ok: boolean;
  summary_result?: {
    parsed_json?: {
      summary?: string;
      key_points?: string[];
      reason?: string;
    };
  };
  matched_message_count?: number;
  [key: string]: unknown;
};

export default function App() {
  const [groupId, setGroupId] = useState("983577967");
  const [timeRange, setTimeRange] = useState<[Dayjs | null, Dayjs | null]>([
    dayjs("2026-04-30T15:26:00"),
    dayjs("2026-04-30T15:45:00"),
  ]);

  // pipeline 结果
  const [result, setResult] = useState<PipelineResponse | null>(null);
  const [loadingPipeline, setLoadingPipeline] = useState(false);

  // 当前选中的 cluster 及 summary
  const [selectedCluster, setSelectedCluster] = useState<Cluster | null>(null);
  const [selectedClassification, setSelectedClassification] =
    useState<Classification | null>(null);
  const [selectedTagResult, setSelectedTagResult] = useState<TagResult | null>(null);

  const [summaryText, setSummaryText] = useState("");
  const [summaryKeyPoints, setSummaryKeyPoints] = useState<string[]>([]);
  const [summaryReason, setSummaryReason] = useState("");
  const [loadingSummary, setLoadingSummary] = useState(false);

  async function handleRunPipeline() {
    if (!groupId || !timeRange[0] || !timeRange[1]) {
      message.warning("请先填写群号和时间范围");
      return;
    }

    setLoadingPipeline(true);

    // 重新分析时，先清空右侧详情
    setSelectedCluster(null);
    setSelectedClassification(null);
    setSelectedTagResult(null);
    setSummaryText("");
    setSummaryKeyPoints([]);
    setSummaryReason("");

    try {
      const response = await fetch("http://127.0.0.1:8000/process/pipeline", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          group_id: groupId,
          start_time: timeRange[0].format("YYYY-MM-DDTHH:mm:ss"),
          end_time: timeRange[1].format("YYYY-MM-DDTHH:mm:ss"),
        }),
      });

      if (!response.ok) {
        throw new Error(`请求失败，状态码: ${response.status}`);
      }

      const data: PipelineResponse = await response.json();
      setResult(data);
      message.success("pipeline 请求成功");
    } catch (error) {
      console.error(error);
      message.error("pipeline 请求失败，请看浏览器控制台报错");
    } finally {
      setLoadingPipeline(false);
    }
  }

  async function handleViewSummary(cluster: Cluster) {
    if (!groupId || !timeRange[0] || !timeRange[1]) {
      message.warning("缺少群号或时间范围");
      return;
    }

    const classification = getClassification(cluster.cluster_id);
    const tagResult = getTagResult(cluster.cluster_id);

    setSelectedCluster(cluster);
    setSelectedClassification(classification);
    setSelectedTagResult(tagResult);

    setLoadingSummary(true);
    setSummaryText("");
    setSummaryKeyPoints([]);
    setSummaryReason("");

    try {
      const response = await fetch("http://127.0.0.1:8000/process/cluster-summary", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          group_id: groupId,
          start_time: timeRange[0].format("YYYY-MM-DDTHH:mm:ss"),
          end_time: timeRange[1].format("YYYY-MM-DDTHH:mm:ss"),
          cluster: {
            message_ids: cluster.message_ids,
            reason: cluster.reason,
            slice_ids: cluster.slice_ids,
            cluster_id: cluster.cluster_id,
          },
          classification: classification
            ? {
                category: classification.category,
                reason: classification.reason,
                cluster_id: classification.cluster_id,
              }
            : null,
          tag: tagResult
            ? {
                tags: tagResult.tags,
                reason: tagResult.reason,
                cluster_id: tagResult.cluster_id,
              }
            : null,
        }),
      });

      if (!response.ok) {
        throw new Error(`summary 请求失败，状态码: ${response.status}`);
      }

      const data: SummaryResponse = await response.json();

      if (!data.ok) {
        throw new Error("summary 接口返回 ok=false");
      }

      const parsed = data.summary_result?.parsed_json;
      setSummaryText(parsed?.summary ?? "");
      setSummaryKeyPoints(parsed?.key_points ?? []);
      setSummaryReason(parsed?.reason ?? "");

      message.success("summary 生成成功");
    } catch (error) {
      console.error(error);
      message.error("summary 请求失败，请看浏览器控制台报错");
    } finally {
      setLoadingSummary(false);
    }
  }

  const clusters = useMemo(() => {
    return result?.cluster_result?.parsed_json?.clusters ?? [];
  }, [result]);

  const classifications = useMemo(() => {
    return result?.classification_result?.parsed_json?.cluster_classification_results ?? [];
  }, [result]);

  const tags = useMemo(() => {
    return result?.tag_result?.parsed_json?.cluster_tag_results ?? [];
  }, [result]);

  function getClassification(clusterId?: string) {
    return classifications.find((item) => item.cluster_id === clusterId) ?? null;
  }

  function getTagResult(clusterId?: string) {
    return tags.find((item) => item.cluster_id === clusterId) ?? null;
  }

  return (
    <div style={{ maxWidth: 1280, margin: "0 auto", padding: 24 }}>
      <Title level={2}>QQ群聊分析 - 前端第三步</Title>

      <Card title="分析条件" style={{ marginBottom: 24 }}>
        <Form layout="vertical">
          <Form.Item label="群号">
            <Input
              value={groupId}
              onChange={(e) => setGroupId(e.target.value)}
              placeholder="请输入群号"
            />
          </Form.Item>

          <Form.Item label="时间范围">
            <DatePicker.RangePicker
              showTime
              style={{ width: "100%" }}
              value={timeRange}
              onChange={(values) =>
                setTimeRange((values as [Dayjs | null, Dayjs | null]) ?? [null, null])
              }
            />
          </Form.Item>

          <Button type="primary" onClick={handleRunPipeline} loading={loadingPipeline}>
            开始分析
          </Button>
        </Form>
      </Card>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        <Card title="Cluster 列表">
          {loadingPipeline ? (
            <Spin />
          ) : clusters.length === 0 ? (
            <Text type="secondary">这里会显示 /process/pipeline 返回的 cluster 列表</Text>
          ) : (
            <List
              dataSource={clusters}
              renderItem={(cluster, index) => {
                const classification = getClassification(cluster.cluster_id);
                const tagResult = getTagResult(cluster.cluster_id);

                return (
                  <List.Item>
                    <Card style={{ width: "100%" }}>
                      <Space direction="vertical" style={{ width: "100%" }} size="small">
                        <Text strong>
                          Cluster {index + 1}
                          {cluster.cluster_id ? `（${cluster.cluster_id}）` : ""}
                        </Text>

                        <div>
                          <Text strong>说明：</Text>
                          <Paragraph style={{ marginBottom: 0 }}>
                            {cluster.reason ?? "无说明"}
                          </Paragraph>
                        </div>

                        <div>
                          <Text strong>分类：</Text>
                          {classification?.category ? (
                            <Tag color="blue">{classification.category}</Tag>
                          ) : (
                            <Text type="secondary">暂无分类</Text>
                          )}
                        </div>

                        <div>
                          <Text strong>标签：</Text>
                          {tagResult?.tags && tagResult.tags.length > 0 ? (
                            tagResult.tags.map((tag) => <Tag key={tag}>{tag}</Tag>)
                          ) : (
                            <Text type="secondary">暂无标签</Text>
                          )}
                        </div>

                        <div>
                          <Text strong>消息数：</Text>
                          <Text>{cluster.message_ids.length}</Text>
                        </div>

                        <div>
                          <Text strong>slice_ids：</Text>
                          {cluster.slice_ids && cluster.slice_ids.length > 0 ? (
                            cluster.slice_ids.map((sliceId) => <Tag key={sliceId}>{sliceId}</Tag>)
                          ) : (
                            <Text type="secondary">暂无</Text>
                          )}
                        </div>

                        <Button onClick={() => handleViewSummary(cluster)} loading={loadingSummary && selectedCluster?.cluster_id === cluster.cluster_id}>
                          查看详细总结
                        </Button>
                      </Space>
                    </Card>
                  </List.Item>
                );
              }}
            />
          )}
        </Card>

        <Card title="Summary 详情">
          {!selectedCluster ? (
            <Text type="secondary">请先在左侧点击一个 cluster 的“查看详细总结”</Text>
          ) : loadingSummary ? (
            <Spin />
          ) : (
            <Space direction="vertical" style={{ width: "100%" }} size="middle">
              <div>
                <Text strong>当前 Cluster：</Text>
                <Text>
                  {selectedCluster.cluster_id ?? "未命名"}
                </Text>
              </div>

              <div>
                <Text strong>分类：</Text>
                {selectedClassification?.category ? (
                  <Tag color="blue">{selectedClassification.category}</Tag>
                ) : (
                  <Text type="secondary">暂无分类</Text>
                )}
              </div>

              <div>
                <Text strong>标签：</Text>
                {selectedTagResult?.tags && selectedTagResult.tags.length > 0 ? (
                  selectedTagResult.tags.map((tag) => <Tag key={tag}>{tag}</Tag>)
                ) : (
                  <Text type="secondary">暂无标签</Text>
                )}
              </div>

              <Divider style={{ margin: "8px 0" }} />

              <div>
                <Text strong>总结：</Text>
                <Paragraph style={{ marginTop: 8 }}>
                  {summaryText || "暂无总结内容"}
                </Paragraph>
              </div>

              <div>
                <Text strong>关键点：</Text>
                {summaryKeyPoints.length > 0 ? (
                  <List
                    size="small"
                    bordered
                    dataSource={summaryKeyPoints}
                    renderItem={(item) => <List.Item>{item}</List.Item>}
                    style={{ marginTop: 8 }}
                  />
                ) : (
                  <Paragraph style={{ marginTop: 8 }}>
                    暂无关键点
                  </Paragraph>
                )}
              </div>

              <div>
                <Text strong>总结说明：</Text>
                <Paragraph style={{ marginTop: 8 }}>
                  {summaryReason || "无"}
                </Paragraph>
              </div>
            </Space>
          )}
        </Card>
      </div>
    </div>
  );
}