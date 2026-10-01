using System;
using System.Data;

namespace WebApplication1
{
    public partial class StudentList : System.Web.UI.Page
    {
        protected void Page_Load(object sender, EventArgs e)
        {
            if (!IsPostBack)
            {
                BindGrid(GetSampleData());
            }
        }

        private DataTable GetSampleData()
        {
            DataTable dt = new DataTable();
            dt.Columns.Add("Roll Number");
            dt.Columns.Add("Student Name");
            dt.Columns.Add("Branch");
            dt.Columns.Add("Email");

            dt.Rows.Add("21B81A0501", "A. Rahul", "CSE", "rahul.a@cvr.ac.in");
            dt.Rows.Add("22B81A1204", "B. Sneha", "IT", "sneha.b@cvr.ac.in");
            dt.Rows.Add("22B81A6615", "K. Vikram", "CSM", "vikram.k@cvr.ac.in");
            dt.Rows.Add("23B81A0410", "M. Divya", "ECE", "divya.m@cvr.ac.in");
            return dt;
        }

        private void BindGrid(DataTable dt)
        {
            gvStudents.DataSource = dt;
            gvStudents.DataBind();
        }

        protected void btnFilter_Click(object sender, EventArgs e)
        {
            DataTable dt = GetSampleData();
            string query = txtSearch.Text.Trim().ToLower();

            if (!string.IsNullOrEmpty(query))
            {
                DataView dv = new DataView(dt);
                dv.RowFilter = "[Student Name] LIKE '%" + query + "%' OR [Branch] LIKE '%" + query + "%'";
                gvStudents.DataSource = dv;
            }
            else
            {
                gvStudents.DataSource = dt;
            }
            gvStudents.DataBind();
        }

        protected void btnReset_Click(object sender, EventArgs e)
        {
            txtSearch.Text = "";
            BindGrid(GetSampleData());
        }
    }
}