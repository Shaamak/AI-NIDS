// Sample UNSW-NB15 network flow data for quick testing.
// These represent realistic network packet features from the dataset schema.
export const SAMPLE_FLOWS = [
  {
    name: "Normal HTTP Traffic",
    features: {
      dur: 0.121478, proto: "tcp", service: "http", state: "FIN",
      spkts: 6, dpkts: 4, sbytes: 954, dbytes: 1680, rate: 82.4,
      sttl: 254, dttl: 252, sload: 62936.39, dload: 110886.31,
      sloss: 0, dloss: 0, sinpkt: 14.46, dinpkt: 31.35,
      sjit: 3.14, djit: 8.19, swin: 255, stcpb: 3704478450,
      dtcpb: 2824697265, dwin: 255, tcprtt: 0.014, synack: 0.007,
      ackdat: 0.007, smean: 159, dmean: 420, trans_depth: 1,
      response_body_len: 1382, ct_srv_src: 2, ct_state_ttl: 2,
      ct_dst_ltm: 1, ct_src_dport_ltm: 1, ct_dst_sport_ltm: 1,
      ct_dst_src_ltm: 2, is_ftp_login: 0, ct_ftp_cmd: 0,
      ct_flw_http_mthd: 2, ct_src_ltm: 1, ct_srv_dst: 2,
      is_sm_ips_ports: 0
    }
  },
  {
    name: "Suspicious DoS Pattern",
    features: {
      dur: 0.000011, proto: "tcp", service: "-", state: "INT",
      spkts: 4, dpkts: 0, sbytes: 220, dbytes: 0, rate: 363636.36,
      sttl: 31, dttl: 0, sload: 160000000, dload: 0,
      sloss: 4, dloss: 0, sinpkt: 0.003, dinpkt: 0,
      sjit: 0, djit: 0, swin: 0, stcpb: 0,
      dtcpb: 0, dwin: 0, tcprtt: 0, synack: 0,
      ackdat: 0, smean: 55, dmean: 0, trans_depth: 0,
      response_body_len: 0, ct_srv_src: 18, ct_state_ttl: 1,
      ct_dst_ltm: 17, ct_src_dport_ltm: 3, ct_dst_sport_ltm: 15,
      ct_dst_src_ltm: 19, is_ftp_login: 0, ct_ftp_cmd: 0,
      ct_flw_http_mthd: 0, ct_src_ltm: 18, ct_srv_dst: 17,
      is_sm_ips_ports: 0
    }
  },
  {
    name: "Reconnaissance Scan",
    features: {
      dur: 0.00182, proto: "tcp", service: "-", state: "CON",
      spkts: 2, dpkts: 2, sbytes: 128, dbytes: 128, rate: 2197.8,
      sttl: 62, dttl: 60, sload: 562637.36, dload: 562637.36,
      sloss: 0, dloss: 0, sinpkt: 0.911, dinpkt: 0.911,
      sjit: 0, djit: 0, swin: 255, stcpb: 1947622310,
      dtcpb: 2568127190, dwin: 255, tcprtt: 0.001, synack: 0.001,
      ackdat: 0, smean: 64, dmean: 64, trans_depth: 0,
      response_body_len: 0, ct_srv_src: 14, ct_state_ttl: 3,
      ct_dst_ltm: 8, ct_src_dport_ltm: 2, ct_dst_sport_ltm: 3,
      ct_dst_src_ltm: 9, is_ftp_login: 0, ct_ftp_cmd: 0,
      ct_flw_http_mthd: 0, ct_src_ltm: 14, ct_srv_dst: 8,
      is_sm_ips_ports: 0
    }
  }
];
