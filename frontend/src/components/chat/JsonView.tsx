import type React from "react";

interface JsonViewProps {
  data: object;
}

const JsonView: React.FC<JsonViewProps> = ({ data }) => {
  return (
    <pre className="bg-neutral-900 p-3 rounded-lg overflow-x-auto font-mono text-xs my-3">
      {JSON.stringify(data, null, 2)}
    </pre>
  );
};

export default JsonView;
